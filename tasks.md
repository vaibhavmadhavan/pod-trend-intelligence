# POD Trend Intelligence — Build Tasks

Status key: `[ ]` todo · `[x]` done · `[-]` blocked

---

## 0. Project Scaffolding

- [x] Create `requirements.txt` with all dependencies (pytrends, praw, selenium, beautifulsoup4, pandas, scikit-learn, xgboost, prophet, bertopic, openai, ollama, langchain, chromadb, streamlit, jinja2, joblib, python-dotenv, pytest, flake8)
- [x] Create `.env.example` with all required keys (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `OPENAI_API_KEY`, `LLM_PROVIDER`)
- [x] Create `.gitignore` (exclude `.env`, `data/`, `models/`, `logs/`, `__pycache__/`, `data/chroma/`)
- [x] Create `config.py` with full constants block (KEYWORDS ~30-50, SUBREDDITS, HOLIDAY_DATES, all thresholds and paths from system design §11)
- [x] Create directory skeleton: `db/migrations/`, `modules/templates/`, `dashboard/`, `models/`, `data/`, `logs/`, `tests/`
- [x] Add `__init__.py` to `modules/`, `dashboard/`, `tests/`

---

## 1. Database Layer (`db/`) — Supabase (PostgreSQL)

- [x] Write `db/migrations/001_initial_schema.sql` — PostgreSQL DDL for `schema_version`, `keyword_runs` (with all columns + UNIQUE constraint + index), `reddit_posts` (with UNIQUE on post_url + index), `llm_outputs` (with UNIQUE on run_date+keyword); use `SERIAL PRIMARY KEY` not `AUTOINCREMENT`
- [x] Write `db/migrations/002_add_is_viable_etsy_listings_niche_themes.sql` — PostgreSQL DDL for `etsy_listings` and `niche_themes` tables + their indexes; `ALTER TABLE keyword_runs ADD COLUMN IF NOT EXISTS is_viable INTEGER DEFAULT 1`
- [x] Write `db/schema.sql` — combined DDL reference (union of all migrations, for documentation)
- [x] Write `db/database.py`:
  - [x] `get_connection()` — reads `DATABASE_URL` from env; returns `psycopg2` connection (`autocommit=False`)
  - [x] `run_migrations()` — bootstraps `schema_version`, globs + sorts `migrations/*.sql`, applies unapplied files in a transaction, records each in `schema_version`
  - [x] `retry_with_backoff(fn, max_attempts=3)` — shared HTTP retry utility with 2s/4s/8s waits
  - [x] `insert_or_ignore()` — `INSERT INTO ... ON CONFLICT DO NOTHING` with `%s` placeholders
  - [x] `update_row()` — `UPDATE ... SET` with `%s` placeholders
  - [x] `fetch_df()` — `pd.read_sql_query` wrapper; returns empty DataFrame on no rows
- [x] Add `psycopg2-binary==2.9.9` to `requirements.txt`
- [x] Add `DATABASE_URL=postgresql://...` to `.env.example`
- [x] Remove `DB_PATH` from `config.py`; keep `MIGRATIONS_DIR`

---

## 2. Module 1 — Data Collection (`modules/collector.py`)

- [ ] `fetch_google_trends(keyword: str) -> float | None` — pulls latest interest score via pytrends; returns None on failure; adds inter-request delay
- [ ] `fetch_reddit_posts(keyword: str, subreddits: list[str], run_date: date) -> list[RedditPost]` — queries PRAW for hot/top posts matching keyword in each subreddit; captures title, upvotes, comment_count, post_url
- [ ] `fetch_etsy_listings(keyword: str, run_date: date, limit: int = 50) -> list[EtsyListing]` — scrapes Etsy search results with BeautifulSoup/Selenium; captures title, price, review_count, listing_age_days, url; enforces 2s rate limit between requests
- [ ] `run_collection(keywords: list[str], run_date: date) -> CollectionResult` — orchestrates all three sub-fetchers per keyword; wraps each in try/except; writes results to `keyword_runs` (partial), `reddit_posts`, `etsy_listings`; returns `CollectionResult(keywords_processed, rows_inserted, errors)`
- [ ] Define `RedditPost`, `EtsyListing`, `CollectionResult` dataclasses
- [ ] JSON-structured logging for each sub-task (module, keyword, level, message, run_date)

---

## 3. Module 2 — Feature Engineering (`modules/features.py`)

- [ ] `compute_features(run_date: date) -> pd.DataFrame`:
  - [ ] `trend_slope` — linear regression slope over last 4 weekly `trend_score` values per keyword
  - [ ] `growth_rate_mom` — `(current - last_month) / last_month × 100`; NULL-safe
  - [ ] `reddit_volume` — count of `reddit_posts` rows for this keyword in current week
  - [ ] `reddit_virality` — avg upvotes of matching reddit posts this week
  - [ ] `etsy_demand` — avg `review_count` of top 20 `etsy_listings` for keyword on run_date
  - [ ] `competition_score` — `etsy_competition / etsy_demand`; handle division-by-zero
  - [ ] `seasonality_flag` — binary: is any HOLIDAY_DATE within SEASONALITY_WINDOW_DAYS of run_date?
  - [ ] UPDATE `keyword_runs` with all computed columns for run_date
  - [ ] Log WARNING (not raise) for any feature that cannot be computed due to insufficient history

---

## 4. Module 3 — ML Opportunity Scoring (`modules/scorer.py`)

- [ ] `score_niches(df: pd.DataFrame, retrain: bool = False) -> pd.DataFrame`:
  - [ ] Heuristic scorer: `clamp(50 + 30×trend_slope_norm - 20×competition_score_norm + 10×seasonality_flag, 0, 100)` using min-max normalisation across current run
  - [ ] XGBoost inference path: load `models/xgb_model.joblib`, predict `opportunity_score`
  - [ ] Retrain path (triggered by `retrain=True` or auto every 30 days): fit XGBoost on full `keyword_runs` history, save model, log feature importances to `logs/feature_importance_YYYYMMDD.json`
  - [ ] Selection logic: use heuristic if `COUNT(DISTINCT run_date) < MIN_HISTORY_DAYS_FOR_ML`; use XGBoost otherwise
  - [ ] Fallback to heuristic if model file is corrupt (log WARNING, delete bad file)
  - [ ] Log active scoring mode (`heuristic` or `xgboost`) on every call
  - [ ] UPDATE `keyword_runs.opportunity_score` for run_date

---

## 5. Module 4 — Trend Forecasting (`modules/forecaster.py`)

- [ ] `forecast_trends(keywords: list[str]) -> dict[str, str]`:
  - [ ] For each keyword: read full `(run_date, trend_score)` time series from `keyword_runs`
  - [ ] Fit Prophet model; generate 30-day forward forecast
  - [ ] Compute forecast slope; classify as Rising (slope > 0.5), Peaking (-0.5 to 0.5), Fading (< -0.5), Unknown (< 4 data points)
  - [ ] Set `is_viable = 0` for Fading; `is_viable = 1` for all others
  - [ ] UPDATE `keyword_runs` (trend_direction, is_viable) for run_date
  - [ ] Return `{"keyword": "Rising|Peaking|Fading|Unknown", ...}`

---

## 6. Module 5 — NLP Theme Clustering (`modules/clusterer.py`)

- [ ] `cluster_themes(keyword: str, run_date: date) -> list[Theme]`:
  - [ ] Read `etsy_listings.title` for (keyword, run_date)
  - [ ] If < 10 titles: return single Theme labelled "general", skip clustering
  - [ ] Primary path: BERTopic — fit on titles, extract top 5 topic labels + representative examples
  - [ ] Fallback path: TF-IDF + K-Means (k=5) if BERTopic import fails (log WARNING)
  - [ ] Gap detection: flag `is_gap = 1` if `cluster_size < 0.1 × total_listings_for_keyword`
  - [ ] INSERT rows into `niche_themes`
  - [ ] Define `Theme` dataclass: `label: str, size: int, examples: list[str], is_gap: bool`

---

## 7. Module 6 — LLM Copy Generation (`modules/llm.py`)

- [ ] Create `modules/templates/llm_prompt_v1.j2` — Jinja2 template matching system design §8 (keyword, scores, themes with GAP markers, JSON output spec)
- [ ] Define `NicheContext` and `LLMOutput` dataclasses
- [ ] `generate_copy(keyword: str, context: NicheContext) -> LLMOutput | None`:
  - [ ] Idempotency check: skip if `llm_outputs` already has row for (today, keyword)
  - [ ] Render Jinja2 template with context
  - [ ] Call OpenAI GPT-4o; parse JSON response into `LLMOutput`
  - [ ] On OpenAI `RateLimitError` or `APIError`: auto-switch to Ollama (log WARNING)
  - [ ] On Ollama failure: log CRITICAL, return None
  - [ ] INSERT into `llm_outputs` (including `model_used` and `prompt_version`)
  - [ ] Provider selection reads `LLM_PROVIDER` from config; version reads `LLM_PROMPT_VERSION`

---

## 8. Pipeline Orchestrator (`pipeline.py`)

- [ ] `run_pipeline(retrain: bool = False) -> PipelineResult`:
  - [ ] Startup: validate all required `.env` keys present (`DATABASE_URL`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `OPENAI_API_KEY`); raise `EnvironmentError` with list of missing keys if not
  - [ ] Call `db.run_migrations()`, open connection
  - [ ] Module 1 → 2 → 3 → 4 in order; pass results through
  - [ ] Filter: `SELECT keywords WHERE is_viable = 1` after Module 4
  - [ ] Module 5 → 6 for each viable keyword
  - [ ] RAG update: load documents from `keyword_runs`, `reddit_posts`, `llm_outputs`; upsert into ChromaDB at `data/chroma/` using LangChain
  - [ ] Write structured JSON log to `logs/pipeline_YYYYMMDD.log`
  - [ ] Return `PipelineResult(run_date, keywords_processed, viable_keywords, llm_outputs_generated, duration_seconds, errors)`
- [ ] Define `PipelineResult` dataclass

---

## 9. Streamlit Dashboard (`dashboard/app.py`)

- [ ] **Sidebar**:
  - [ ] "Run Pipeline" button — calls `subprocess.run(["python", "pipeline.py"], check=True)` with `st.spinner`
  - [ ] "Last updated" timestamp — reads `MAX(run_date)` from `keyword_runs`

- [ ] **Page 1 — Trending Now**:
  - [ ] Ranked table: `SELECT * FROM keyword_runs WHERE run_date = (latest) ORDER BY opportunity_score DESC`
  - [ ] Display columns: keyword, opportunity_score, trend_direction badge (↑ Rising / → Peaking / ↓ Fading), competition level label (Low/Medium/High derived from competition_score)

- [ ] **Page 2 — Niche Deep Dive**:
  - [ ] Keyword selector dropdown
  - [ ] Google Trends chart: `trend_score` over time for selected keyword
  - [ ] Reddit volume chart: `reddit_volume` over time
  - [ ] BERTopic theme list with gap markers
  - [ ] Full LLM copy output (strategic insight, recommended angle, slogans, Etsy title, tags)

- [ ] **Page 3 — AI Chat**:
  - [ ] LangChain `ConversationalRetrievalChain` backed by ChromaDB vector store at `data/chroma/`
  - [ ] Chat input; response grounded in scraped dataset
  - [ ] Conversation history maintained in `st.session_state`

- [ ] Ensure app launches with single `streamlit run dashboard/app.py` command

---

## 10. Tests (`tests/`)

- [ ] `test_features.py`:
  - [ ] `trend_slope` computed correctly from 4 data points
  - [ ] `growth_rate_mom` returns None (not crash) when last-month data missing
  - [ ] `competition_score` handles zero etsy_demand without raising
  - [ ] `seasonality_flag` = 1 when holiday is within 6 weeks, 0 otherwise

- [ ] `test_scorer.py`:
  - [ ] Heuristic score is clamped to [0, 100]
  - [ ] Score increases with higher trend_slope and lower competition_score
  - [ ] Falls back to heuristic when model file is absent

- [ ] `test_forecaster.py`:
  - [ ] Keywords with < 4 data points get `Unknown` / `is_viable = 1`
  - [ ] Positive slope → Rising; negative slope → Fading

- [ ] `test_clusterer.py`:
  - [ ] Returns single "general" theme for < 10 titles
  - [ ] `is_gap` flagged correctly (cluster_size < 10% of total)

- [ ] `test_llm.py`:
  - [ ] Idempotency: second call with same (date, keyword) does not hit API
  - [ ] Returns None (does not raise) when both providers fail

---

## 11. Non-Functional / Cross-Cutting

- [ ] Confirm flake8 passes with zero errors across all modules (`flake8 .`)
- [ ] All secrets loaded from `.env` — grep codebase for hardcoded credentials before first commit
- [ ] Each module independently runnable: add `if __name__ == "__main__":` demo block to each module file
- [ ] Verify full pipeline completes in < 15 minutes on a 30-keyword list (time one end-to-end run)
- [ ] Etsy scraping enforces minimum 2s delay between requests (assert in test or code review)
- [ ] Log rotation: keep only 30 days of log files (implement in `pipeline.py` startup or a `utils.py` helper)
