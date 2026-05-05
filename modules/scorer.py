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
    log = logging.getLogger("scorer")
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
        "module": "scorer",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))

# Feature columns
FEATURE_COLS = [
    "trend_score",
    "trend_slope",
    "growth_rate_mom",
    "reddit_volume",
    "reddit_virality",
    "etsy_demand",
    "competition_score",
    "seasonality_flag",
]
# Heuristic scorer
def _heuristic_score(row: pd.Series) -> float:
    score = 50.0
    # Trend score: Google's 0-100 interest, centered at 50
    if pd.notna(row.get("trend_score")):
        score += (float(row["trend_score"]) - 50.0) * 0.3
    # Trend slope: rising = good, falling = bad
    if pd.notna(row.get("trend_slope")):
        slope_norm = float(np.clip(row["trend_slope"] / 5.0, -1.0, 1.0))
        score += slope_norm * 20.0
    # Month-over-month growth
    if pd.notna(row.get("growth_rate_mom")):
        growth_norm = float(np.clip(row["growth_rate_mom"] / 50.0, -1.0, 1.0))
        score += growth_norm * 10.0
    # Competition: lower ratio = more room to win
    if pd.notna(row.get("competition_score")):
        comp = float(np.clip(row["competition_score"], 0.0, 1.0))
        score -= comp * 15.0
    # Etsy demand: higher favourites/reviews = proven buyers exist
    if pd.notna(row.get("etsy_demand")):
        demand_norm = float(np.clip(row["etsy_demand"] / 500.0, 0.0, 1.0))
        score += demand_norm * 10.0
    # Reddit signals
    if pd.notna(row.get("reddit_volume")):
        score += min(float(row["reddit_volume"]), 10.0) * 0.5
    if pd.notna(row.get("reddit_virality")):
        score += float(np.clip(row["reddit_virality"] / 1000.0, 0.0, 1.0)) * 5.0
    # Seasonality bonus
    score += float(row.get("seasonality_flag") or 0) * 10.0
    return float(np.clip(score, 0.0, 100.0))

# Feature matrix builder — fills NULLs with safe defaults
def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].copy()
    X["trend_score"] = X["trend_score"].fillna(50.0)
    X["trend_slope"] = X["trend_slope"].fillna(0.0)
    X["growth_rate_mom"] = X["growth_rate_mom"].fillna(0.0)
    X["reddit_volume"] = X["reddit_volume"].fillna(0.0)
    X["reddit_virality"] = X["reddit_virality"].fillna(0.0)
    demand_median = X["etsy_demand"].median()
    X["etsy_demand"] = X["etsy_demand"].fillna(demand_median if pd.notna(demand_median) else 0.0)
    X["competition_score"] = X["competition_score"].fillna(0.5)
    X["seasonality_flag"] = X["seasonality_flag"].fillna(0.0)
    return X.astype(float).values

# XGBoost helpers
def _get_history_days(conn) -> int:
    df = fetch_df(conn, "SELECT MIN(run_date) AS earliest FROM keyword_runs")
    if df.empty or pd.isna(df["earliest"].iloc[0]):
        return 0
    earliest = pd.to_datetime(df["earliest"].iloc[0]).date()
    return (date.today() - earliest).days

def _should_retrain(model_path: Path) -> bool:
    if not model_path.exists():
        return True
    mtime = datetime.fromtimestamp(model_path.stat().st_mtime).date()
    return (date.today() - mtime).days >= config.MODEL_RETRAIN_INTERVAL_DAYS

def _train_xgboost(conn, model_path: Path):
    try:
        import joblib
        from xgboost import XGBRegressor
    except ImportError as exc:
        _log("ERROR", f"XGBoost/joblib not installed: {exc}")
        return None
    df = fetch_df(conn, f"""
        SELECT {', '.join(FEATURE_COLS)}
        FROM keyword_runs
        WHERE trend_score IS NOT NULL
    """)
    if len(df) < 10:
        _log("WARNING", "Not enough rows to train XGBoost — need at least 10", rows=len(df))
        return None
    X = _build_feature_matrix(df)
    y = np.array([_heuristic_score(row) for _, row in df.iterrows()])
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in model.feature_importances_]))
    imp_path = config.LOG_DIR / f"feature_importance_{date.today().strftime('%Y%m%d')}.json"
    imp_path.write_text(json.dumps(importances, indent=2))
    _log("INFO", "XGBoost trained and saved", rows=len(df), model_path=str(model_path), importances=importances)
    return model

def _score_with_xgboost(df: pd.DataFrame, conn) -> tuple[np.ndarray, str]:
    try:
        import joblib
    except ImportError:
        scores = np.array([_heuristic_score(row) for _, row in df.iterrows()])
        return scores, "heuristic_fallback"

    model_path = config.MODEL_PATH

    if _should_retrain(model_path):
        model = _train_xgboost(conn, model_path)
        if model is None:
            scores = np.array([_heuristic_score(row) for _, row in df.iterrows()])
            return scores, "heuristic_fallback"
    else:
        model = joblib.load(model_path)
    X = _build_feature_matrix(df)
    scores = np.clip(model.predict(X), 0.0, 100.0)
    return scores, "xgboost"

# Main entry point
def compute_scores(run_date: date) -> pd.DataFrame:
    conn = get_connection()

    df = fetch_df(conn, f"""
        SELECT keyword, {', '.join(FEATURE_COLS)}
        FROM keyword_runs
        WHERE run_date = %s
    """, (run_date,))
    if df.empty:
        _log("WARNING", "No rows to score — run collector first", run_date=str(run_date))
        conn.close()
        return df
    history_days = _get_history_days(conn)
    if history_days >= config.MIN_HISTORY_DAYS_FOR_ML:
        scores, mode = _score_with_xgboost(df, conn)
    else:
        scores = np.array([_heuristic_score(row) for _, row in df.iterrows()])
        mode = "heuristic"
        _log("INFO",
             f"Using heuristic scorer ({history_days} days of history, need {config.MIN_HISTORY_DAYS_FOR_ML} for XGBoost)",
             run_date=str(run_date))
    df["opportunity_score"] = np.round(scores, 2)
    for _, row in df.iterrows():
        update_row(
            conn,
            "keyword_runs",
            updates={"opportunity_score": float(row["opportunity_score"])},
            where={"run_date": run_date, "keyword": row["keyword"]},
        )
    _log("INFO", f"Scored {len(df)} keywords",
         mode=mode,
         run_date=str(run_date),
         scores={row["keyword"]: float(row["opportunity_score"]) for _, row in df.iterrows()})
    conn.close()
    return df[["keyword", "opportunity_score"]].sort_values("opportunity_score", ascending=False)

if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    df = compute_scores(target)
    if df.empty:
        print("No rows — run collector first for this date.")
    else:
        print(df.to_string(index=False))
