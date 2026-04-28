from pathlib import Path

# ---------------------------------------------------------------------------
# Keywords (~30-50 POD niches to track)
# ---------------------------------------------------------------------------
KEYWORDS = [
    # Pet niches
    "dog mom", "cat dad", "dog dad", "cat mom", "crazy dog lady",
    "bunny mom", "fish dad", "reptile mom",
    # Profession niches
    "nurse life", "teacher appreciation", "teacher life", "firefighter wife",
    "police wife", "paramedic life", "social worker life",
    # Hobby niches
    "hiking lover", "gym motivation", "plant mom", "fishing dad",
    "camping lover", "yoga life", "gardening mom", "coffee lover",
    "book lover", "wine mom", "beer dad",
    # Family / identity niches
    "boy mom", "girl dad", "twin mom", "dog grandma",
    "proud dad", "soccer mom", "baseball mom",
    # Seasonal / gift niches
    "best dad ever", "best mom ever", "blessed grandma",
    "retirement gift", "graduation gift",
]

# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------
SUBREDDITS = [
    "funny",
    "aww",
    "fitness",
    "mildlyinteresting",
    "Etsy",
    "printondemand",
]

# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
HOLIDAY_DATES = {
    "Valentine's Day": "2027-02-14",
    "Mother's Day":    "2027-05-11",
    "Father's Day":    "2027-06-15",
    "Halloween":       "2026-10-31",
    "Christmas":       "2026-12-25",
}
SEASONALITY_WINDOW_DAYS = 42  # 6 weeks

# ---------------------------------------------------------------------------
# Etsy scraping
# ---------------------------------------------------------------------------
ETSY_DELAY_SECONDS = 2.0
ETSY_LISTINGS_PER_KEYWORD = 50

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER = "openai"       # overridden by .env LLM_PROVIDER
LLM_MODEL_OPENAI = "gpt-4o"
LLM_MODEL_OLLAMA = "llama3"
LLM_PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
MODEL_RETRAIN_INTERVAL_DAYS = 30
MIN_HISTORY_DAYS_FOR_ML = 90

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
PROPHET_FORECAST_DAYS = 30
TREND_SLOPE_RISING_THRESHOLD = 0.5
TREND_SLOPE_FADING_THRESHOLD = -0.5

# ---------------------------------------------------------------------------
# Paths (relative to project root; pipeline.py resolves to absolute)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# DATABASE_URL is loaded from .env — not a local path
MODEL_PATH = BASE_DIR / "models" / "xgb_model.joblib"
CHROMA_PATH = BASE_DIR / "data" / "chroma"
LOG_DIR = BASE_DIR / "logs"
TEMPLATE_DIR = BASE_DIR / "modules" / "templates"
MIGRATIONS_DIR = BASE_DIR / "db" / "migrations"
