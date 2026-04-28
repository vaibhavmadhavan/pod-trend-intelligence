CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

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
    UNIQUE(run_date, keyword)
);

CREATE INDEX IF NOT EXISTS idx_keyword_runs_keyword_date
    ON keyword_runs(keyword, run_date);

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

CREATE TABLE IF NOT EXISTS llm_outputs (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE    NOT NULL,
    keyword             TEXT    NOT NULL,
    strategic_insight   TEXT,
    recommended_angle   TEXT,
    slogans             TEXT,
    etsy_title          TEXT,
    etsy_tags           TEXT,
    model_used          TEXT,
    prompt_version      TEXT,
    UNIQUE(run_date, keyword)
);
