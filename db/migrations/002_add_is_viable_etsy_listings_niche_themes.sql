ALTER TABLE keyword_runs ADD COLUMN IF NOT EXISTS is_viable INTEGER DEFAULT 1;

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

CREATE TABLE IF NOT EXISTS niche_themes (
    id                      SERIAL PRIMARY KEY,
    run_date                DATE    NOT NULL,
    keyword                 TEXT    NOT NULL,
    theme_label             TEXT    NOT NULL,
    cluster_size            INTEGER,
    representative_titles   TEXT,
    is_gap                  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_niche_themes_keyword_date
    ON niche_themes(keyword, run_date);
