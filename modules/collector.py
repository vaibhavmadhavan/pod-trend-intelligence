import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from pytrends.request import TrendReq

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from db.database import get_connection, insert_or_ignore, retry_with_backoff, update_row

load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    log_path = config.LOG_DIR / f"pipeline_{date.today().strftime('%Y%m%d')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("collector")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(handler)
        log.addHandler(console)
    return log


_logger = _setup_logger()


def _log(level: str, message: str, **extra) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "module": "collector",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RedditPost:
    keyword: str
    subreddit: str
    post_title: str
    upvotes: int
    comment_count: int
    post_url: str
    scraped_date: date


@dataclass
class EtsyListing:
    keyword: str
    run_date: date
    title: str
    price: float | None
    review_count: int | None
    listing_age_days: int | None
    url: str


@dataclass
class CollectionResult:
    keywords_processed: int
    rows_inserted: int
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------

_pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))


def fetch_google_trends(keyword: str) -> float | None:
    try:
        _pytrends.build_payload([keyword], cat=0, timeframe="today 3-m")
        df = _pytrends.interest_over_time()
        if df.empty or keyword not in df.columns:
            _log("WARNING", "No data returned", keyword=keyword, source="google_trends")
            return None
        time.sleep(5)
        return float(df[keyword].iloc[-1])
    except Exception as exc:
        _log("ERROR", f"Fetch failed: {exc}", keyword=keyword, source="google_trends")
        return None


# ---------------------------------------------------------------------------
# Reddit (requires REDDIT_CLIENT_ID in .env — graceful stub until registered)
# ---------------------------------------------------------------------------

def fetch_reddit_posts(
    keyword: str, subreddits: list[str], run_date: date
) -> list[RedditPost]:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    if not client_id:
        _log("WARNING", "REDDIT_CLIENT_ID not set — skipping Reddit", keyword=keyword)
        return []

    try:
        import praw

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )

        posts: dict[str, RedditPost] = {}
        for sub in subreddits:
            for mode in ("hot", "top"):
                try:
                    sr = reddit.subreddit(sub)
                    feed = sr.hot(limit=25) if mode == "hot" else sr.top(limit=25, time_filter="week")
                    for post in feed:
                        if keyword.lower() in post.title.lower() and post.url not in posts:
                            posts[post.url] = RedditPost(
                                keyword=keyword,
                                subreddit=sub,
                                post_title=post.title,
                                upvotes=post.score,
                                comment_count=post.num_comments,
                                post_url=post.url,
                                scraped_date=run_date,
                            )
                except Exception as exc:
                    _log("WARNING", f"Subreddit fetch failed: {exc}", keyword=keyword, subreddit=sub)

        return list(posts.values())

    except Exception as exc:
        _log("CRITICAL", f"Reddit collection failed: {exc}", keyword=keyword)
        return []


# ---------------------------------------------------------------------------
# Etsy API v3
# ---------------------------------------------------------------------------

_ETSY_BASE = "https://openapi.etsy.com/v3/application"
_ETSY_LISTING_URL = "https://www.etsy.com/listing/{listing_id}"


def _etsy_headers() -> dict[str, str]:
    key = os.environ.get("ETSY_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("ETSY_API_KEY is not set in .env")
    return {"x-api-key": key}


def fetch_etsy_listings(
    keyword: str, run_date: date, limit: int = 50
) -> list[EtsyListing]:
    try:
        headers = _etsy_headers()
    except EnvironmentError as exc:
        _log("ERROR", str(exc), keyword=keyword, source="etsy")
        return []

    listings: list[EtsyListing] = []
    offset = 0
    page_size = min(limit, 100)  # Etsy max per request is 100

    while len(listings) < limit:
        params = {
            "keywords": keyword,
            "limit": min(page_size, limit - len(listings)),
            "offset": offset,
            "sort_on": "score",
            "sort_order": "desc",
        }

        def _do_request(p=params, h=headers):
            resp = requests.get(f"{_ETSY_BASE}/listings/active", params=p, headers=h, timeout=15)
            resp.raise_for_status()
            return resp.json()

        try:
            data = retry_with_backoff(_do_request)
        except Exception as exc:
            _log("ERROR", f"Etsy API request failed: {exc}", keyword=keyword, source="etsy", run_date=str(run_date))
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            listing_id = item.get("listing_id")
            title = (item.get("title") or "").strip()
            if not title or not listing_id:
                continue

            price_data = item.get("price") or {}
            try:
                price = float(price_data.get("amount", 0)) / max(price_data.get("divisor", 1), 1)
            except (TypeError, ValueError):
                price = None

            num_favorers = item.get("num_favorers")
            views = item.get("views")
            # Etsy API doesn't expose review_count on listing search results;
            # use num_favorers as the closest proxy for social proof / demand signal.
            review_count = num_favorers if num_favorers is not None else views

            url = _ETSY_LISTING_URL.format(listing_id=listing_id)

            listings.append(EtsyListing(
                keyword=keyword,
                run_date=run_date,
                title=title[:500],
                price=price,
                review_count=review_count,
                listing_age_days=None,
                url=url,
            ))

        offset += len(results)
        if len(results) < page_size:
            break  # no more pages

        time.sleep(config.ETSY_DELAY_SECONDS)

    _log("INFO", f"Fetched {len(listings)} Etsy listings", keyword=keyword, source="etsy", run_date=str(run_date))
    return listings[:limit]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_collection(keywords: list[str], run_date: date) -> CollectionResult:
    conn = get_connection()
    rows_inserted = 0
    errors: list[str] = []

    for keyword in keywords:
        # Google Trends → partial keyword_runs row
        try:
            trend_score = fetch_google_trends(keyword)
            insert_or_ignore(conn, "keyword_runs", {
                "run_date": run_date,
                "keyword": keyword,
                "trend_score": trend_score,
            })
            rows_inserted += 1
        except Exception as exc:
            msg = f"keyword_runs insert failed: {exc}"
            _log("ERROR", msg, keyword=keyword, run_date=str(run_date))
            errors.append(f"{keyword}: {msg}")

        # Reddit → reddit_posts
        try:
            posts = fetch_reddit_posts(keyword, config.SUBREDDITS, run_date)
            for post in posts:
                insert_or_ignore(conn, "reddit_posts", {
                    "scraped_date": post.scraped_date,
                    "keyword": post.keyword,
                    "subreddit": post.subreddit,
                    "post_title": post.post_title,
                    "upvotes": post.upvotes,
                    "comment_count": post.comment_count,
                    "post_url": post.post_url,
                })
                rows_inserted += 1
        except Exception as exc:
            msg = f"Reddit insert failed: {exc}"
            _log("ERROR", msg, keyword=keyword, run_date=str(run_date))
            errors.append(f"{keyword}: {msg}")

        # Etsy → etsy_listings + update etsy_competition
        try:
            listings = fetch_etsy_listings(keyword, run_date, limit=config.ETSY_LISTINGS_PER_KEYWORD)
            for listing in listings:
                insert_or_ignore(conn, "etsy_listings", {
                    "run_date": listing.run_date,
                    "keyword": listing.keyword,
                    "title": listing.title,
                    "price": listing.price,
                    "review_count": listing.review_count,
                    "listing_age_days": listing.listing_age_days,
                    "url": listing.url,
                })
                rows_inserted += 1
            update_row(
                conn,
                "keyword_runs",
                updates={"etsy_competition": len(listings)},
                where={"run_date": run_date, "keyword": keyword},
            )
        except Exception as exc:
            msg = f"Etsy insert failed: {exc}"
            _log("ERROR", msg, keyword=keyword, run_date=str(run_date))
            errors.append(f"{keyword}: {msg}")

    conn.close()
    return CollectionResult(
        keywords_processed=len(keywords),
        rows_inserted=rows_inserted,
        errors=errors,
    )


if __name__ == "__main__":
    result = run_collection(["dog mom", "cat dad"], date.today())
    print(f"\nKeywords processed : {result.keywords_processed}")
    print(f"Rows inserted      : {result.rows_inserted}")
    if result.errors:
        print(f"Errors             : {result.errors}")
