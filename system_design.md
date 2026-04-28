# POD Trend Intelligence System — System Design

## 1. System Overview

The POD Trend Intelligence System is a single-user pipeline that runs on demand (or on a schedule) to identify high-opportunity print-on-demand niches. Each run collects live data from Google Trends, Reddit, and Etsy; engineers growth and competition features; scores every niche with an ML model; forecasts whether each trend is rising or fading; clusters Etsy listing titles into sub-themes; and sends the viable niches to an LLM that returns strategic insight and ready-to-paste listing copy. Everything is stored in a Supabase-hosted PostgreSQL database that accumulates over time, and a Streamlit dashboard (hosted on Streamlit Community Cloud) surfaces the results with a RAG-powered chat interface. The pipeline runs locally and writes to Supabase over the network. v1 is sequential and single-process — no async, no job queue.

---

## 2. Architecture Diagram

```
 .env ──────────────────┐
 config.py ─────────────┤
                        ▼
                  ┌─────────────┐
                  │ pipeline.py │  ← entry point; orchestrates all modules
                  └──────┬──────┘
                         │
         ┌───────────────▼───────────────┐
         │        MODULE 1               │
         │        collector.py           │
         │  pytrends | PRAW | Selenium   │
         └───────────────┬───────────────┘
                         │ writes
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌───────────────┐ ┌──────────┐ ┌──────────────┐
  │ keyword_runs  │ │reddit_   │ │etsy_listings │
  │ (partial row) │ │posts     │ │              │
  └───────┬───────┘ └──────────┘ └──────────────┘
          │
         ▼
  ┌───────────────┐
  │   MODULE 2    │  reads keyword_runs + history
  │   features.py │  → updates keyword_runs (all feature cols)
  └───────┬───────┘
          │
         ▼
  ┌───────────────┐
  │   MODULE 3    │  reads keyword_runs (full features)
  │   scorer.py   │  → updates opportunity_score
  └───────┬───────┘  → saves/loads models/xgb_model.joblib
          │
         ▼
  ┌───────────────┐
  │   MODULE 4    │  reads keyword_runs history (time series)
  │  forecaster.py│  → updates trend_direction + is_viable
  └───────┬───────┘
          │
          │  pipeline.py filters: is_viable = 1 only
          │
         ▼
  ┌───────────────┐
  │   MODULE 5    │  reads etsy_listings for viable keywords
  │  clusterer.py │  → writes niche_themes
  └───────┬───────┘
          │
         ▼
  ┌───────────────┐
  │   MODULE 6    │  reads keyword_runs + niche_themes
  │    llm.py     │  → writes llm_outputs
  └───────┬───────┘
          │
         ▼
  ┌─────────────────────────┐
  │  RAG updater (pipeline) │  upserts keyword_runs + reddit_posts
  │  LangChain + ChromaDB   │  + llm_outputs into data/chroma/
  └─────────────────────────┘
          │
         ▼
  ┌───────────────┐
  │   MODULE 7    │  reads all tables + ChromaDB
  │ dashboard/    │  displays results, triggers pipeline
  │   app.py      │
  └───────────────┘
```

---

## 3. Module Interface Contracts

### `modules/collector.py`

```python
def run_collection(keywords: list[str], run_date: date) -> CollectionResult:
    ...
```

| | Detail |
|---|---|
| **Reads from** | Google Trends API (pytrends), Reddit API (PRAW), Etsy (Selenium/BS4) |
| **Writes to** | `keyword_runs` (partial: trend_score, etsy_competition), `reddit_posts`, `etsy_listings` |
| **Returns** | `CollectionResult(keywords_processed: int, rows_inserted: int, errors: list[str])` |
| **Fails gracefully by** | Wrapping each sub-task (Google, Reddit, Etsy) in try/except; on failure, logs `ERROR` and continues with remaining keywords/sources. Missing values left as NULL in DB. |

Sub-functions (called internally, also independently testable):

```python
def fetch_google_trends(keyword: str) -> float | None
def fetch_reddit_posts(keyword: str, subreddits: list[str], run_date: date) -> list[RedditPost]
def fetch_etsy_listings(keyword: str, run_date: date, limit: int = 50) -> list[EtsyListing]
```

---

### `modules/features.py`

```python
def compute_features(run_date: date) -> pd.DataFrame:
    ...
```

| | Detail |
|---|---|
| **Reads from** | `keyword_runs` (current + historical rows) |
| **Writes to** | `keyword_runs` (updates: trend_slope, growth_rate_mom, reddit_volume, reddit_virality, etsy_demand, competition_score, seasonality_flag) |
| **Returns** | DataFrame of updated rows for the given `run_date` |
| **Fails gracefully by** | If historical data is insufficient for a feature (e.g., < 4 weeks for trend_slope), writes NULL and logs `WARNING`. Never raises on missing data. |

---

### `modules/scorer.py`

```python
def score_niches(df: pd.DataFrame, retrain: bool = False) -> pd.DataFrame:
    ...
```

| | Detail |
|---|---|
| **Reads from** | `models/xgb_model.joblib` (if exists); `keyword_runs` (full history, for retraining) |
| **Writes to** | `keyword_runs` (updates: opportunity_score); `models/xgb_model.joblib` (on retrain) |
| **Returns** | DataFrame with `opportunity_score` column populated |
| **Fails gracefully by** | Falls back to heuristic scoring if model file missing or retrain fails. Logs which mode is active (`heuristic` or `xgboost`) on every call. |

---

### `modules/forecaster.py`

```python
def forecast_trends(keywords: list[str]) -> dict[str, str]:
    ...
```

| | Detail |
|---|---|
| **Reads from** | `keyword_runs` (historical trend_score time series per keyword) |
| **Writes to** | `keyword_runs` (updates: trend_direction, is_viable) |
| **Returns** | `{"dog mom": "Rising", "cat dad": "Fading", ...}` |
| **Fails gracefully by** | If < 4 historical data points exist for a keyword, sets trend_direction = "Unknown" and is_viable = 1 (don't block early keywords). Logs `WARNING`. |

Classification rules (applied to 30-day Prophet forecast slope):

| Slope | Classification | is_viable |
|-------|---------------|-----------|
| > +0.5 | Rising | 1 |
| −0.5 to +0.5 | Peaking | 1 |
| < −0.5 | Fading | 0 |
| Insufficient data | Unknown | 1 |

---

### `modules/clusterer.py`

```python
def cluster_themes(keyword: str, run_date: date) -> list[Theme]:
    ...
```

| | Detail |
|---|---|
| **Reads from** | `etsy_listings` (titles for the given keyword and run_date) |
| **Writes to** | `niche_themes` |
| **Returns** | `list[Theme(label: str, size: int, examples: list[str], is_gap: bool)]` |
| **Fails gracefully by** | If < 10 titles available, skips BERTopic and returns a single theme labelled "general". If BERTopic fails (import error), falls back to TF-IDF + K-Means with k=5. |

Gap detection: a cluster is flagged `is_gap = 1` if its `cluster_size < 0.1 × total_listings_for_keyword`.

---

### `modules/llm.py`

```python
def generate_copy(keyword: str, context: NicheContext) -> LLMOutput:
    ...
```

```python
@dataclass
class NicheContext:
    opportunity_score: float
    competition_score: float
    trend_direction: str
    themes: list[Theme]

@dataclass
class LLMOutput:
    strategic_insight: str
    recommended_angle: str
    slogans: list[str]        # 5 items
    etsy_title: str           # ≤ 140 chars
    etsy_tags: list[str]      # 10 items, each ≤ 20 chars
    model_used: str
    prompt_version: str
```

| | Detail |
|---|---|
| **Reads from** | `modules/templates/llm_prompt_v1.j2` (Jinja2 template) |
| **Writes to** | `llm_outputs` |
| **Returns** | `LLMOutput` dataclass |
| **Fails gracefully by** | On OpenAI failure → retries with Ollama. If both fail → logs `CRITICAL`, returns None, pipeline skips this keyword's LLM output silently. |

Provider selection: reads `LLM_PROVIDER` from `config.py`; overridden to `"ollama"` automatically on OpenAI `RateLimitError` or `APIError`.

---

### `pipeline.py`

```python
def run_pipeline(retrain: bool = False) -> PipelineResult:
    ...
```

| | Detail |
|---|---|
| **Orchestrates** | Modules 1 → 2 → 3 → 4 → (filter is_viable) → 5 → 6 → RAG update |
| **Returns** | `PipelineResult(run_date, keywords_processed, viable_keywords, llm_outputs_generated, duration_seconds, errors)` |
| **Writes** | Structured JSON log to `logs/pipeline_YYYYMMDD.log` |
| **DB** | Connects to Supabase via `DATABASE_URL`; calls `run_migrations()` on startup |

Startup validation: on import, `pipeline.py` checks that all required `.env` keys are present (`DATABASE_URL`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `OPENAI_API_KEY`). Raises `EnvironmentError` with a clear message listing missing keys before any API calls are made.

---

### `dashboard/app.py`

No function contract — Streamlit app. Key state interactions:

| UI Element | Action |
|---|---|
| Sidebar "Run Pipeline" button | Calls `subprocess.run(["python", "pipeline.py"])` in blocking mode; shows `st.spinner` |
| "Last updated" timestamp | Reads `MAX(run_date)` from `keyword_runs` |
| Page 1 table | `SELECT * FROM keyword_runs WHERE run_date = (latest) ORDER BY opportunity_score DESC` |
| Page 2 charts | `SELECT trend_score, run_date FROM keyword_runs WHERE keyword = ?` (time series) |
| Page 3 chat | LangChain `ConversationalRetrievalChain` over ChromaDB vector store |

---

## 4. Full Database Schema

### `schema_version` (migration tracking)

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
```

### `keyword_runs`

```sql
CREATE TABLE IF NOT EXISTS keyword_runs (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE    NOT NULL,
    keyword             TEXT    NOT NULL,
    trend_score         REAL,
    trend_slope         REAL,
    growth_rate_mom     REAL,
    reddit_volume       INTEGER,
    reddit_virality     REAL,
    etsy_competition    INTEGER,
    etsy_demand         REAL,
    competition_score   REAL,
    seasonality_flag    INTEGER,
    opportunity_score   REAL,
    trend_direction     TEXT,
    is_viable           INTEGER DEFAULT 1,
    UNIQUE(run_date, keyword)
);

CREATE INDEX IF NOT EXISTS idx_keyword_runs_keyword_date
    ON keyword_runs(keyword, run_date);
```

### `reddit_posts`

```sql
CREATE TABLE IF NOT EXISTS reddit_posts (
    id              SERIAL PRIMARY KEY,
    scraped_date    DATE    NOT NULL,
    keyword         TEXT    NOT NULL,
    subreddit       TEXT    NOT NULL,
    post_title      TEXT,
    upvotes         INTEGER,
    comment_count   INTEGER,
    post_url        TEXT,
    UNIQUE(post_url)
);

CREATE INDEX IF NOT EXISTS idx_reddit_keyword_date
    ON reddit_posts(keyword, scraped_date);
```

### `etsy_listings` (new)

```sql
CREATE TABLE IF NOT EXISTS etsy_listings (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE    NOT NULL,
    keyword             TEXT    NOT NULL,
    title               TEXT    NOT NULL,
    price               REAL,
    review_count        INTEGER,
    listing_age_days    INTEGER,
    url                 TEXT,
    UNIQUE(run_date, keyword, url)
);

CREATE INDEX IF NOT EXISTS idx_etsy_listings_keyword_date
    ON etsy_listings(keyword, run_date);
```

### `niche_themes` (new)

```sql
CREATE TABLE IF NOT EXISTS niche_themes (
    id                      SERIAL PRIMARY KEY,
    run_date                DATE    NOT NULL,
    keyword                 TEXT    NOT NULL,
    theme_label             TEXT    NOT NULL,
    cluster_size            INTEGER,
    representative_titles   TEXT,   -- JSON array
    is_gap                  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_niche_themes_keyword_date
    ON niche_themes(keyword, run_date);
```

### `llm_outputs`

```sql
CREATE TABLE IF NOT EXISTS llm_outputs (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE    NOT NULL,
    keyword             TEXT    NOT NULL,
    strategic_insight   TEXT,
    recommended_angle   TEXT,
    slogans             TEXT,   -- JSON array of 5 strings
    etsy_title          TEXT,
    etsy_tags           TEXT,   -- comma-separated, 10 tags
    model_used          TEXT,
    prompt_version      TEXT,
    UNIQUE(run_date, keyword)
);
```

### Migration strategy

- All DDL lives in `db/migrations/001_initial_schema.sql`, `002_add_is_viable.sql`, etc.
- `db/database.py` runs pending migrations on startup by comparing `schema_version` against files in `db/migrations/`
- Migrations are additive only (ADD COLUMN, CREATE TABLE, CREATE INDEX) — no DROP
- PostgreSQL supports `ADD COLUMN IF NOT EXISTS` natively (v9.6+); no workaround needed
- SQL uses PostgreSQL dialect: `SERIAL PRIMARY KEY`, `%s` placeholders, `INSERT ... ON CONFLICT DO NOTHING`

---

## 5. Data Flow — One Pipeline Run (Step by Step)

```
pipeline.py starts
│
├─ 1. Load config.py → keyword list, constants
├─ 2. Validate .env → raise EnvironmentError if missing keys
├─ 3. db/database.py → run pending migrations, open SQLite connection (WAL mode)
│
├─ MODULE 1: collector.py
│   For each keyword:
│   ├─ pytrends → INSERT partial row into keyword_runs (trend_score)
│   ├─ PRAW → INSERT rows into reddit_posts
│   └─ Selenium/BS4 → INSERT rows into etsy_listings
│                   → UPDATE keyword_runs (etsy_competition)
│
├─ MODULE 2: features.py
│   ├─ READ keyword_runs (current + last 4 weeks history per keyword)
│   ├─ Compute: trend_slope, growth_rate_mom, reddit_volume, reddit_virality,
│   │           etsy_demand, competition_score, seasonality_flag
│   └─ UPDATE keyword_runs (all feature columns)
│
├─ MODULE 3: scorer.py
│   ├─ READ keyword_runs (fully featured rows for run_date)
│   ├─ If model exists and history < 90 days → heuristic scoring
│   ├─ If model exists and history ≥ 90 days → XGBoost inference
│   ├─ If retrain=True → retrain XGBoost, save to models/xgb_model.joblib
│   └─ UPDATE keyword_runs (opportunity_score)
│
├─ MODULE 4: forecaster.py
│   ├─ READ keyword_runs (all historical trend_score per keyword)
│   ├─ Fit Prophet per keyword → generate 30-day forecast
│   ├─ Classify slope → Rising / Peaking / Fading / Unknown
│   └─ UPDATE keyword_runs (trend_direction, is_viable)
│
├─ pipeline.py FILTER
│   └─ SELECT keywords WHERE is_viable = 1 → viable_keywords list
│
├─ MODULE 5: clusterer.py
│   For each viable keyword:
│   ├─ READ etsy_listings (titles for keyword + run_date)
│   ├─ Run BERTopic (or TF-IDF + K-Means fallback)
│   ├─ Detect gaps (cluster_size < 10% of total)
│   └─ INSERT rows into niche_themes
│
├─ MODULE 6: llm.py
│   For each viable keyword:
│   ├─ READ keyword_runs (opportunity_score, competition_score, trend_direction)
│   ├─ READ niche_themes (theme labels + examples)
│   ├─ Check if llm_outputs already has row for (today, keyword) → skip if yes
│   ├─ Render Jinja2 template → call OpenAI GPT-4o (or Ollama fallback)
│   └─ INSERT into llm_outputs
│
├─ RAG UPDATE (pipeline.py)
│   ├─ Load documents from reddit_posts, keyword_runs, llm_outputs
│   ├─ Upsert into ChromaDB at data/chroma/ (incremental)
│   └─ Log "RAG index updated: N documents"
│
└─ Log PipelineResult to logs/pipeline_YYYYMMDD.log
```

---

## 6. Error Handling Strategy

### Retry policy (HTTP requests)

```
Attempt 1 → wait 2s → Attempt 2 → wait 4s → Attempt 3 → wait 8s → log CRITICAL, skip keyword
```

Implemented as a shared `retry_with_backoff(fn, max_attempts=3)` utility in `db/database.py` or a shared `utils.py`.

### Per-module behaviour on failure

| Module | Failure scenario | Behaviour |
|--------|-----------------|-----------|
| 1 — Google Trends | API rate-limit / network error | Retry ×3; if all fail, trend_score = NULL, log ERROR, continue to Reddit |
| 1 — Reddit | API credentials invalid | Log CRITICAL once; skip Reddit for entire run |
| 1 — Etsy | HTTP 429 / block detected | Retry with 2× delay; after 3 fails, etsy_* columns = NULL for that keyword |
| 2 — Features | Insufficient history | Write NULL for that feature; log WARNING per keyword |
| 3 — Scorer | Model file corrupt | Delete and fall back to heuristic; log WARNING |
| 4 — Forecaster | < 4 data points | Set Unknown / is_viable = 1; log WARNING |
| 5 — Clusterer | BERTopic import error | Fall back to TF-IDF + K-Means; log WARNING |
| 6 — LLM (OpenAI) | RateLimitError / APIError | Auto-switch to Ollama; log WARNING |
| 6 — LLM (Ollama) | Ollama not running | Skip LLM output for keyword; log CRITICAL |

### Log format

Each log line is a JSON object:

```json
{
  "timestamp": "2026-04-27T14:32:01Z",
  "level": "ERROR",
  "module": "collector",
  "keyword": "dog mom",
  "message": "Etsy request failed after 3 attempts: HTTP 429",
  "run_date": "2026-04-27"
}
```

Log files: `logs/pipeline_YYYYMMDD.log`. Rotate weekly; keep 30 days.

---

## 7. ML Cold Start Strategy

### Phase 1 — Heuristic scoring (days 1–89)

When `COUNT(DISTINCT run_date) < 90` in `keyword_runs`:

```
opportunity_score = clamp(
    50
    + 30 × trend_slope_norm
    - 20 × competition_score_norm
    + 10 × seasonality_flag,
    0, 100
)
```

Where `_norm` = min-max normalised across all keywords in the current run.

This is deterministic and interpretable — a keyword with fast-growing trend, low competition, and an approaching holiday scores near 100.

### Phase 2 — XGBoost (day 90+)

- Training target: `opportunity_score` as computed by heuristic (uses history as ground truth)
- Features: all columns in `keyword_runs` except `id`, `run_date`, `keyword`, `opportunity_score`, `trend_direction`, `is_viable`
- Triggered by: `run_pipeline(retrain=True)` or automatically every 30 days
- Model saved to `models/xgb_model.joblib`; feature importance logged to `logs/feature_importance_YYYYMMDD.json`

---

## 8. Prompt Versioning

Template location: `modules/templates/llm_prompt_v1.j2`

```jinja
You are an expert print-on-demand product strategist.

Niche: {{ keyword }}
Opportunity score: {{ opportunity_score }}/100
Competition score: {{ competition_score }} (lower = less saturated)
Trend direction: {{ trend_direction }}
Top sub-themes on Etsy right now:
{% for theme in themes %}
- {{ theme.label }} ({{ theme.size }} listings){% if theme.is_gap %} ← GAP{% endif %}
{% endfor %}

Return a JSON object with these exact keys:
- strategic_insight: 2-sentence analysis of whether to enter this niche and why
- recommended_angle: one sub-niche angle to differentiate
- slogans: list of 5 product copy ideas (t-shirt / mug ready)
- etsy_title: one SEO-optimised Etsy listing title, max 140 characters
- etsy_tags: list of 10 tags, each max 20 characters
```

Version tracking:
- `config.py`: `LLM_PROMPT_VERSION = "v1"`
- Written to `llm_outputs.prompt_version` on insert
- To update: create `llm_prompt_v2.j2`, set `LLM_PROMPT_VERSION = "v2"`, redeploy

---

## 9. Orchestration Model (v1)

```
pipeline.py
  │
  ├─ Sequential per module (M1 → M2 → M3 → M4 → filter → M5 → M6 → RAG)
  ├─ Sequential per keyword within M1 (no concurrency)
  ├─ No async / no job queue
  └─ Dashboard triggers via: subprocess.run(["python", "pipeline.py"], check=True)
```

Expected runtime breakdown (30 keywords):

| Step | Estimated time |
|------|---------------|
| Module 1 — Google Trends | 2–3 min (rate limits) |
| Module 1 — Reddit | 1 min |
| Module 1 — Etsy (2s/page × 50 listings × 30 keywords) | 5 min |
| Modules 2–4 | < 1 min |
| Module 5 (BERTopic) | 1–2 min |
| Module 6 (LLM, 20 viable keywords) | 2–3 min |
| RAG update | < 1 min |
| **Total** | **~12–15 min** |

v2 parallelism path: `concurrent.futures.ThreadPoolExecutor(max_workers=3)` for Module 1 sub-tasks per keyword; reduces Module 1 from 8 min to ~3 min.

---

## 10. Caching Strategy

| Layer | What is cached | Where | TTL |
|-------|---------------|-------|-----|
| Google Trends | Raw API response per `(keyword, date)` | In-memory dict (within a run) | Single run |
| LLM outputs | Full `llm_outputs` row | Supabase (PostgreSQL) | Until next run_date (idempotent re-run check) |
| ChromaDB | Full vector index | `data/chroma/` on disk | Persistent; upserted each run |
| Etsy / Reddit | Not cached | — | Always fresh |

Idempotent re-run: if `pipeline.py` is run twice on the same day, Module 6 checks `SELECT 1 FROM llm_outputs WHERE run_date=today AND keyword=?` and skips the API call if a row exists.

---

## 11. Configuration (`config.py`)

```python
# Keywords
KEYWORDS = [
    "dog mom", "cat dad", "nurse life", "teacher appreciation",
    "hiking lover", "gym motivation", "plant mom", "fishing dad",
    # ... ~30-50 total
]

# Reddit
SUBREDDITS = ["funny", "aww", "fitness", "mildlyinteresting", "Etsy", "printondemand"]

# Seasonality
HOLIDAY_DATES = {
    "Valentine's Day": "2027-02-14",
    "Mother's Day":    "2027-05-11",
    "Father's Day":    "2027-06-15",
    "Halloween":       "2026-10-31",
    "Christmas":       "2026-12-25",
}
SEASONALITY_WINDOW_DAYS = 42  # 6 weeks

# Etsy scraping
ETSY_DELAY_SECONDS = 2.0
ETSY_LISTINGS_PER_KEYWORD = 50

# LLM
LLM_PROVIDER = "openai"        # "openai" | "ollama"
LLM_MODEL_OPENAI = "gpt-4o"
LLM_MODEL_OLLAMA = "llama3"
LLM_PROMPT_VERSION = "v1"

# ML
MODEL_RETRAIN_INTERVAL_DAYS = 30
MIN_HISTORY_DAYS_FOR_ML = 90

# Forecasting
PROPHET_FORECAST_DAYS = 30
TREND_SLOPE_RISING_THRESHOLD = 0.5
TREND_SLOPE_FADING_THRESHOLD = -0.5

# Database — loaded from .env, not hardcoded
# DATABASE_URL = "postgresql://..." (set in .env)

# Paths
MODEL_PATH = "models/xgb_model.joblib"
CHROMA_PATH = "data/chroma"
LOG_DIR = "logs"
TEMPLATE_DIR = "modules/templates"
```

---

## 12. Scalability Path

| Concern | v1 (now) | v2 (future) |
|---------|----------|-------------|
| Database | Supabase free tier (PostgreSQL) | Supabase Pro or self-hosted PostgreSQL + connection pool |
| Scheduling | Manual or `schedule` Python lib | cron / Prefect / Airflow |
| Module 1 concurrency | Sequential per keyword | `ThreadPoolExecutor(max_workers=3)` |
| Dashboard hosting | Streamlit Community Cloud (free) | Custom domain + Docker + nginx |
| Keyword list | Static in `config.py` | User-editable via dashboard sidebar |
| Model validation | Manual inspection of feature importances | `actual_outcomes` feedback table; RMSE computed quarterly |
| Multi-user | N/A | Auth layer + per-user keyword sets |

Supabase free → Pro migration: connection string change in `.env` only; schema and all queries are unchanged.

---

## 13. Project File Structure (Final)

```
pod-trend-intelligence/
├── requirements.md
├── system_design.md              ← this file
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
├── db/
│   ├── database.py               ← connection, migration runner, query helpers
│   ├── schema.sql                ← combined DDL for reference
│   └── migrations/
│       ├── 001_initial_schema.sql
│       └── 002_add_is_viable_etsy_listings_niche_themes.sql
├── modules/
│   ├── __init__.py
│   ├── collector.py
│   ├── features.py
│   ├── scorer.py
│   ├── forecaster.py
│   ├── clusterer.py
│   ├── llm.py
│   └── templates/
│       └── llm_prompt_v1.j2
├── pipeline.py
├── dashboard/
│   ├── __init__.py
│   └── app.py
├── models/
│   └── xgb_model.joblib          ← (generated at runtime)
├── data/
│   └── chroma/                   ← (generated at runtime; database is on Supabase)
├── logs/                         ← (generated at runtime)
└── tests/
    ├── __init__.py
    ├── test_features.py
    ├── test_scorer.py
    ├── test_forecaster.py
    ├── test_clusterer.py
    └── test_llm.py
```

---

## 14. Verification Checklist

Before writing any code, confirm this document is complete:

- [x] Every module from requirements.md has an interface contract (Section 3)
- [x] All 5 tables defined with full DDL and indexes (Section 4)
- [x] `etsy_listings` and `niche_themes` added beyond original requirements
- [x] Every NF requirement addressed:
  - NF1 (< 15 min): Section 9 runtime breakdown
  - NF2 (secrets in .env): Section 3 pipeline.py startup validation
  - NF3 (modules independently runnable): Section 3 interface contracts
  - NF4 (schema versioning): Section 4 migration strategy
  - NF5 (Etsy rate limiting): Section 6 retry policy
  - NF6 (graceful degradation): Section 6 per-module failure table
  - NF7 (flake8): enforced via pre-commit (noted in requirements.txt)
  - NF8 (pytest): Section 13 test file list
- [x] All 14 design ambiguities from exploration resolved (Sections 6–11)
- [x] Document is self-contained — does not require reading requirements.md to understand
