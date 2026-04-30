import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from db.database import fetch_df, get_connection, update_row

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
    log = logging.getLogger("features")
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
        "module": "features",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _compute_trend_slope(keyword: str, history: pd.DataFrame) -> float | None:
    """Linear regression slope over up to the last 4 trend_score data points."""
    last4 = history.tail(4)
    if len(last4) < 2:
        _log("WARNING", "Insufficient history for trend_slope", keyword=keyword, points=len(last4))
        return None
    if len(last4) < 4:
        _log("WARNING", "Fewer than 4 points for trend_slope — using available data", keyword=keyword, points=len(last4))
    x = np.arange(len(last4), dtype=float)
    y = last4["trend_score"].astype(float).values
    return float(np.polyfit(x, y, 1)[0])


def _compute_growth_rate_mom(
    keyword: str,
    history: pd.DataFrame,
    run_date: date,
    current: float | None,
) -> float | None:
    """Percent change vs the closest run to 30 days ago (±14 day window)."""
    if current is None:
        return None
    target = pd.Timestamp(run_date - timedelta(days=30))
    lo = target - timedelta(days=14)
    hi = target + timedelta(days=14)
    run_ts = pd.Timestamp(run_date)
    window = history[
        (history["run_date"] >= lo)
        & (history["run_date"] <= hi)
        & (history["run_date"] < run_ts)
    ]
    if window.empty:
        _log("WARNING", "No data for growth_rate_mom", keyword=keyword)
        return None
    last_month = float(window.iloc[-1]["trend_score"])
    if last_month == 0:
        return None
    return (current - last_month) / last_month * 100


def _seasonality_flag(run_date: date) -> int:
    """1 if any configured holiday falls within SEASONALITY_WINDOW_DAYS of run_date."""
    for holiday_str in config.HOLIDAY_DATES.values():
        holiday = date.fromisoformat(holiday_str)
        if abs((holiday - run_date).days) <= config.SEASONALITY_WINDOW_DAYS:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_features(run_date: date) -> pd.DataFrame:
    """Compute all feature columns for every keyword that ran on run_date.

    Writes results back to keyword_runs and returns a DataFrame summary.
    """
    conn = get_connection()
    try:
        # --- Base rows for this run ---
        base_df = fetch_df(
            conn,
            "SELECT keyword, trend_score, etsy_competition "
            "FROM keyword_runs WHERE run_date = %s",
            (run_date,),
        )
        if base_df.empty:
            _log("WARNING", "No keyword_runs rows found for run_date", run_date=str(run_date))
            return pd.DataFrame()

        keywords = base_df["keyword"].tolist()

        # --- Historical trend scores (for slope + MoM) ---
        history_df = fetch_df(
            conn,
            """
            SELECT keyword, run_date, trend_score
            FROM keyword_runs
            WHERE keyword = ANY(%s)
              AND run_date <= %s
              AND trend_score IS NOT NULL
            ORDER BY keyword, run_date
            """,
            (keywords, run_date),
        )
        history_df["run_date"] = pd.to_datetime(history_df["run_date"])

        # --- Reddit posts this week ---
        week_start = run_date - timedelta(days=7)
        reddit_df = fetch_df(
            conn,
            """
            SELECT keyword, upvotes
            FROM reddit_posts
            WHERE keyword = ANY(%s)
              AND scraped_date >= %s
              AND scraped_date <= %s
            """,
            (keywords, week_start, run_date),
        )

        # --- Etsy listings for run_date ---
        etsy_df = fetch_df(
            conn,
            """
            SELECT keyword, review_count
            FROM etsy_listings
            WHERE keyword = ANY(%s)
              AND run_date = %s
              AND review_count IS NOT NULL
            ORDER BY keyword, review_count DESC
            """,
            (keywords, run_date),
        )

        season_flag = _seasonality_flag(run_date)
        results: list[dict] = []

        for _, base_row in base_df.iterrows():
            keyword = base_row["keyword"]
            current_score = (
                float(base_row["trend_score"])
                if base_row["trend_score"] is not None
                else None
            )
            etsy_competition = (
                int(base_row["etsy_competition"])
                if base_row["etsy_competition"] is not None
                else None
            )

            kw_hist = history_df[history_df["keyword"] == keyword].sort_values("run_date")
            kw_reddit = reddit_df[reddit_df["keyword"] == keyword]
            kw_etsy = etsy_df[etsy_df["keyword"] == keyword].head(20)

            trend_slope = _compute_trend_slope(keyword, kw_hist)
            growth_rate_mom = _compute_growth_rate_mom(keyword, kw_hist, run_date, current_score)
            reddit_volume = len(kw_reddit)
            reddit_virality = (
                float(kw_reddit["upvotes"].mean()) if not kw_reddit.empty else None
            )
            etsy_demand = (
                float(kw_etsy["review_count"].mean()) if not kw_etsy.empty else None
            )
            competition_score = (
                etsy_competition / etsy_demand
                if etsy_demand and etsy_demand > 0 and etsy_competition is not None
                else None
            )

            updates = {
                "trend_slope": trend_slope,
                "growth_rate_mom": growth_rate_mom,
                "reddit_volume": reddit_volume,
                "reddit_virality": reddit_virality,
                "etsy_demand": etsy_demand,
                "competition_score": competition_score,
                "seasonality_flag": season_flag,
            }

            update_row(conn, "keyword_runs", updates, {"run_date": run_date, "keyword": keyword})
            _log("INFO", "Features computed", keyword=keyword, run_date=str(run_date),
                 trend_slope=trend_slope, growth_rate_mom=growth_rate_mom,
                 reddit_volume=reddit_volume, etsy_demand=etsy_demand,
                 competition_score=competition_score, seasonality_flag=season_flag)

            results.append({"keyword": keyword, **updates})

        return pd.DataFrame(results)

    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    df = compute_features(target)
    if df.empty:
        print("No rows — run collector first for this date.")
    else:
        print(df.to_string(index=False))
