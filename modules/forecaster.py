import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from db.database import fetch_df, get_connection, update_row

# Logging
def _setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    log_path = config.LOG_DIR / f"pipeline_{date.today().strftime('%Y%m%d')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("forecaster")
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
        "module": "forecaster",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))

# Constants
MIN_POINTS_FOR_PROPHET = 4

# Classification
def _classify_slope(slope_per_week: float) -> str:
    if slope_per_week > config.TREND_SLOPE_RISING_THRESHOLD:
        return "Rising"
    if slope_per_week < config.TREND_SLOPE_FADING_THRESHOLD:
        return "Fading"
    return "Peaking"
# Per-keyword forecast
def _forecast_keyword(keyword: str, history: pd.DataFrame) -> tuple[str, float | None]:
    """Fits Prophet on history and returns (direction, slope_per_week).
    Falls back to trend_slope column if Prophet is unavailable.
    Returns ('Unknown', None) when there is not enough data."""
    if len(history) < MIN_POINTS_FOR_PROPHET:
        _log("INFO", f"Not enough history ({len(history)} points) — classifying as Unknown",
             keyword=keyword, points=len(history))
        return "Unknown", None
    #Prophet path
    try:
        import logging as _std_logging
        _std_logging.getLogger("cmdstanpy").setLevel(_std_logging.ERROR)
        _std_logging.getLogger("prophet").setLevel(_std_logging.ERROR)

        from prophet import Prophet

        df_p = pd.DataFrame({
            "ds": pd.to_datetime(history["run_date"]),
            "y": history["trend_score"].astype(float),
        }).dropna()
        if len(df_p) < MIN_POINTS_FOR_PROPHET:
            return "Unknown", None
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_p)
        future = model.make_future_dataframe(periods=config.PROPHET_FORECAST_DAYS)
        forecast = model.predict(future)
        # Slope over the forecast window (yhat is per-day; convert to per-week)
        fcast = forecast.tail(config.PROPHET_FORECAST_DAYS)
        x = np.arange(len(fcast), dtype=float)
        slope_per_day = float(np.polyfit(x, fcast["yhat"].values, 1)[0])
        slope_per_week = slope_per_day * 7
        direction = _classify_slope(slope_per_week)
        return direction, round(slope_per_week, 4)
    except ImportError:
        _log("WARNING", "Prophet not installed — falling back to trend_slope column", keyword=keyword)
    #Fallback: use precomputed trend_slope from features module
    recent_slope = history["trend_slope"].dropna()
    if recent_slope.empty:
        return "Unknown", None
    slope = float(recent_slope.iloc[-1])
    return _classify_slope(slope), round(slope, 4)
# Main entry point
def compute_forecasts(run_date: date) -> dict[str, str]:
    """Forecast trend direction for every keyword in run_date.
    Updates trend_direction and is_viable in keyword_runs.
    Returns {keyword: direction}."""
    conn = get_connection()
    today_df = fetch_df(
        conn,
        "SELECT keyword FROM keyword_runs WHERE run_date = %s",
        (run_date,),
    )
    if today_df.empty:
        _log("WARNING", "No keywords found for run_date — run collector first",
             run_date=str(run_date))
        conn.close()
        return {}
    results: dict[str, str] = {}
    for keyword in today_df["keyword"].tolist():
        history = fetch_df(conn, """
            SELECT run_date, trend_score, trend_slope
            FROM keyword_runs
            WHERE keyword = %s
              AND trend_score IS NOT NULL
            ORDER BY run_date ASC
        """, (keyword,))
        direction, slope = _forecast_keyword(keyword, history)
        is_viable = 0 if direction == "Fading" else 1
        update_row(
            conn,
            "keyword_runs",
            updates={"trend_direction": direction, "is_viable": is_viable},
            where={"run_date": run_date, "keyword": keyword},
        )
        results[keyword] = direction
        _log("INFO", f"Forecast: {direction}",
             keyword=keyword,
             slope_per_week=slope,
             is_viable=is_viable,
             run_date=str(run_date))
    _log("INFO", f"Forecasted {len(results)} keywords",
         run_date=str(run_date),
         summary=results)
    conn.close()
    return results

if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    results = compute_forecasts(target)
    if not results:
        print("No results — run collector first for this date.")
    else:
        for keyword, direction in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {keyword:<30} {direction}")
