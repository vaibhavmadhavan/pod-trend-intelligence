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

def fetch_google_trends(keyword: str) -> float | None:
    api_key = os.environ.get("SERPAPI_KEY", "").strip()
    if not api_key:
        _log("ERROR", "SERPAPI_KEY not set — skipping Google Trends", keyword=keyword)
        return None
    try:
        resp = requests.get(
            _SERPAPI_URL,
            params={
                "engine": "google_trends",
                "q": keyword,
                "data_type": "TIMESERIES",
                "api_key": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        timeline = data.get("interest_over_time", {}).get("timeline_data", [])
        if not timeline:
            _log("WARNING", "No trend data returned", keyword=keyword, source="google_trends")
            return None
        last_values = timeline[-1].get("values", [])
        score = float(last_values[0]["extracted_value"]) if last_values else None
        _log("INFO", f"Trend score: {score}", keyword=keyword, source="google_trends")
        return score
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
# Etsy listings via SerpApi (Google Shopping filtered to Etsy)
# ---------------------------------------------------------------------------

_SERPAPI_URL = "https://serpapi.com/search"


def fetch_etsy_listings(
    keyword: str, run_date: date, limit: int = 50
) -> list[EtsyListing]:
    api_key = os.environ.get("SERPAPI_KEY", "").strip()
    if not api_key:
        _log("ERROR", "SERPAPI_KEY not set — skipping Etsy", keyword=keyword, source="serpapi")
        return []

    params = {
        "engine": "google_shopping",
        "q": f"{keyword} etsy",
        "api_key": api_key,
        "num": 100,
        "hl": "en",
        "gl": "us",
    }

    try:
        resp = requests.get(_SERPAPI_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:500] if exc.response is not None else "no body"
        _log("ERROR", f"SerpApi HTTP error: {exc} — {body}", keyword=keyword, source="serpapi", run_date=str(run_date))
        return []
    except Exception as exc:
        _log("ERROR", f"SerpApi request failed: {exc}", keyword=keyword, source="serpapi", run_date=str(run_date))
        return []

    listings: list[EtsyListing] = []
    for item in data.get("shopping_results", []):
        title = (item.get("title") or "").strip()
        link = (item.get("link") or item.get("product_link") or "").strip()
        if not title or not link:
            continue

        price_raw = item.get("price")
        try:
            price = float(str(price_raw).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            price = None

        review_count = item.get("reviews")

        listings.append(EtsyListing(
            keyword=keyword,
            run_date=run_date,
            title=title[:500],
            price=price,
            review_count=review_count,
            listing_age_days=None,
            url=link,
        ))

        if len(listings) >= limit:
            break

    _log("INFO", f"Fetched {len(listings)} Etsy listings via SerpApi", keyword=keyword, source="serpapi", run_date=str(run_date))
    return listings


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
