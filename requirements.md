# POD Trend Intelligence System — Requirements Document

## 1. Project Overview

An end-to-end data science and AI pipeline that collects live internet data, identifies trending print-on-demand (POD) product niches, scores their commercial opportunity, forecasts whether each trend is rising or fading, and auto-generates ready-to-use Etsy listing copy — to drive real product decisions for a live POD business.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Identify emerging POD niches before they become saturated |
| G2 | Score each niche on opportunity (demand vs. competition) using ML |
| G3 | Forecast whether a trend is rising, peaking, or fading |
| G4 | Auto-generate actionable Etsy listing copy (title, tags, slogans) |
| G5 | Display everything in a self-serve dashboard with a chat interface |
| G6 | Build a proprietary historical dataset over time (not one-shot scrapes) |

---

## 3. Functional Requirements

### 3.1 Data Collection (Module 1)

| ID | Requirement |
|----|-------------|
| F1.1 | Pull Google Trends interest scores (0–100) for a predefined keyword list via `pytrends` |
| F1.2 | Scrape hot/top posts from POD-relevant subreddits (r/funny, r/aww, r/fitness, r/mildlyinteresting) via `PRAW`, capturing title, upvotes, and comment count |
| F1.3 | Scrape Etsy search results per keyword via `BeautifulSoup`/`Selenium`, capturing listing count, average review count, price range, and listing age |
| F1.4 | Append all collected records to a persistent database with a `run_date` timestamp — never overwrite existing rows |
| F1.5 | Use Supabase (PostgreSQL) as the database for all environments, including local development; connect via `DATABASE_URL` from `.env` |
| F1.6 | Log each pipeline run (start time, keywords processed, rows inserted, errors) to a file |

### 3.2 Feature Engineering (Module 2)

| ID | Requirement |
|----|-------------|
| F2.1 | Compute `trend_score`: latest Google Trends interest score |
| F2.2 | Compute `trend_slope`: linear regression slope over the last 4 weekly data points |
| F2.3 | Compute `growth_rate_MoM`: `(current_score − last_month_score) / last_month_score × 100` |
| F2.4 | Compute `reddit_volume`: number of Reddit posts mentioning the keyword this week |
| F2.5 | Compute `reddit_virality`: average upvotes of matching Reddit posts |
| F2.6 | Compute `etsy_competition`: total listing count returned by Etsy for the keyword |
| F2.7 | Compute `etsy_demand`: average review count of the top 20 Etsy listings |
| F2.8 | Compute `competition_score`: `etsy_competition / etsy_demand` (low = opportunity) |
| F2.9 | Compute `seasonality_flag`: binary — is a major gifting holiday within 6 weeks? (Valentine's Day, Mother's Day, Christmas, Halloween) |

### 3.3 ML Opportunity Scoring (Module 3)

| ID | Requirement |
|----|-------------|
| F3.1 | Train an XGBoost regressor on historical feature data to output `opportunity_score` (0–100) |
| F3.2 | Bootstrap training with synthetic labels when historical data is insufficient (< 3 months) |
| F3.3 | Persist trained model to disk with `joblib`; reload on each run without retraining |
| F3.4 | Expose a `retrain` flag that triggers full retraining when invoked |
| F3.5 | Log feature importances after each training run |

### 3.4 Trend Forecasting (Module 4)

| ID | Requirement |
|----|-------------|
| F4.1 | Fit a Facebook Prophet model (or PyTorch LSTM as a stretch goal) to each keyword's Google Trends time series |
| F4.2 | Generate a 30-day forward forecast per keyword |
| F4.3 | Classify each keyword as **Rising**, **Peaking**, or **Fading** based on the sign and magnitude of the forecast slope |
| F4.4 | Filter out **Fading** keywords before passing niches to the recommendation layer |

### 3.5 NLP Theme Clustering (Module 5)

| ID | Requirement |
|----|-------------|
| F5.1 | Collect Etsy product titles for a given keyword (top 50–100 listings) |
| F5.2 | Apply BERTopic (or TF-IDF + K-Means fallback) to cluster titles into recurring sub-themes |
| F5.3 | Return the top 5 theme labels per keyword with representative example titles |
| F5.4 | Identify under-represented theme gaps (small cluster size relative to niche volume) |

### 3.6 LLM Insight & Copy Generation (Module 6)

| ID | Requirement |
|----|-------------|
| F6.1 | Accept as input: keyword, opportunity_score, competition_score, trend direction, top BERTopic themes |
| F6.2 | Return a 2-sentence strategic insight ("Should I enter this niche and why?") |
| F6.3 | Return a recommended sub-niche angle for differentiation |
| F6.4 | Return 5 slogan/product copy ideas (t-shirt / mug ready) |
| F6.5 | Return 1 SEO-optimised Etsy listing title (≤ 140 characters) |
| F6.6 | Return 10 Etsy tags (each ≤ 20 characters, comma-separated) |
| F6.7 | Support OpenAI GPT-4o as primary provider; support Ollama (LLaMA 3) as a zero-cost fallback |
| F6.8 | Prompt must be versioned (stored as a template string, not hardcoded inline) |

### 3.7 Streamlit Dashboard (Module 7)

| ID | Requirement |
|----|-------------|
| F7.1 | **Page 1 — Trending Now**: ranked table of top niches with opportunity score, trend direction badge (↑ → ↓), competition level label |
| F7.2 | **Page 2 — Niche Deep Dive**: select a keyword; display Google Trends chart, Reddit volume chart, BERTopic theme list, and full LLM copy output |
| F7.3 | **Page 3 — AI Chat**: LangChain + ChromaDB RAG interface; user can ask "What should I list this week?" and receive answers grounded in the scraped dataset |
| F7.4 | Sidebar: "Run Pipeline" button that triggers the full collection → feature → score → forecast → LLM chain; displays last-updated timestamp |
| F7.5 | Dashboard must be runnable locally with a single `streamlit run` command |

---

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF1 | Full pipeline run must complete within 15 minutes on a standard laptop |
| NF2 | All secrets (API keys, `DATABASE_URL`) must be loaded from a `.env` file; never hardcoded |
| NF3 | Each module must be independently runnable (importable and callable in isolation) |
| NF4 | Database schema must be versioned; migrations handled by script, not by dropping tables |
| NF5 | Scraping must include polite rate-limiting (minimum 2-second delay between Etsy requests) |
| NF6 | The system must degrade gracefully: if one data source fails (e.g., Reddit API down), the pipeline continues with available data and logs the failure |
| NF7 | Code must pass `flake8` linting with no errors |
| NF8 | Core feature engineering and scoring logic must have unit tests (`pytest`) |

---

## 5. Data Model (Initial Schema)

### `keyword_runs` table
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | auto-increment |
| run_date | DATE | date of pipeline run |
| keyword | TEXT | e.g., "dog mom" |
| trend_score | REAL | Google Trends 0–100 |
| trend_slope | REAL | 4-week linear slope |
| growth_rate_mom | REAL | month-over-month % |
| reddit_volume | INTEGER | weekly post count |
| reddit_virality | REAL | avg upvotes |
| etsy_competition | INTEGER | total listing count |
| etsy_demand | REAL | avg review count (top 20) |
| competition_score | REAL | competition / demand |
| seasonality_flag | INTEGER | 0 or 1 |
| opportunity_score | REAL | ML model output |
| trend_direction | TEXT | Rising / Peaking / Fading |

### `llm_outputs` table
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | auto-increment |
| run_date | DATE | |
| keyword | TEXT | |
| strategic_insight | TEXT | |
| recommended_angle | TEXT | |
| slogans | TEXT | JSON array |
| etsy_title | TEXT | |
| etsy_tags | TEXT | comma-separated |
| model_used | TEXT | e.g., "gpt-4o" |

### `reddit_posts` table
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | auto-increment |
| scraped_date | DATE | |
| keyword | TEXT | matched keyword |
| subreddit | TEXT | |
| post_title | TEXT | |
| upvotes | INTEGER | |
| comment_count | INTEGER | |
| post_url | TEXT | |

---

## 6. External Dependencies & APIs

| Service | Library | Auth Required | Notes |
|---------|---------|---------------|-------|
| Google Trends | `pytrends` | No | Rate-limited; add delays between calls |
| Reddit | `PRAW` | Yes — Reddit API credentials | Free tier sufficient |
| Etsy | `BeautifulSoup` + `requests` / `Selenium` | No (scraping) | Respect robots.txt; use delays |
| OpenAI | `openai` | Yes — API key | GPT-4o preferred |
| Ollama (LLaMA 3) | `ollama` Python client | No (local) | Fallback; requires local install |
| Supabase | `psycopg2-binary` | Yes — `DATABASE_URL` connection string | Free tier; 500MB storage; connect via standard PostgreSQL wire protocol |

---

## 7. Project Structure

```
pod-trend-intelligence/
├── requirements.md              ← this file
├── requirements.txt             ← Python dependencies
├── .env.example                 ← template for secrets
├── config.py                    ← keyword list, holiday dates, constants
├── db/
│   ├── schema.sql               ← table definitions
│   └── database.py              ← connection + query helpers
├── modules/
│   ├── collector.py             ← Module 1: data collection
│   ├── features.py              ← Module 2: feature engineering
│   ├── scorer.py                ← Module 3: ML scoring
│   ├── forecaster.py            ← Module 4: Prophet forecasting
│   ├── clusterer.py             ← Module 5: BERTopic clustering
│   └── llm.py                   ← Module 6: LLM copy generation
├── pipeline.py                  ← orchestrates all modules end-to-end
├── dashboard/
│   └── app.py                   ← Module 7: Streamlit app
├── models/                      ← saved joblib model files
├── data/                        ← local SQLite database file
├── logs/                        ← pipeline run logs
└── tests/
    ├── test_features.py
    └── test_scorer.py
```

---

## 8. Out of Scope (v1)

- Automatic Etsy listing creation via the Etsy API
- Multi-user / hosted deployment (v1 is local only)
- Image generation for product mockups
- Competitor seller analysis (individual shop tracking)
- Email / Slack alerts for new trending niches

---

## 9. Success Criteria

| Criterion | Target |
|-----------|--------|
| Pipeline runs end-to-end without manual intervention | Yes |
| At least 1 niche identified per run with `opportunity_score` > 70 | Yes |
| LLM output used to create at least 1 real Etsy listing | Yes |
| Historical data accumulates across runs (rows increase each run) | Yes |
| Dashboard loads and displays current run data in < 5 seconds | Yes |
