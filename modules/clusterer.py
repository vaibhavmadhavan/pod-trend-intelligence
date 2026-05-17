import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from db.database import fetch_df, get_connection
# Logging
def _setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    log_path = config.LOG_DIR / f"pipeline_{date.today().strftime('%Y%m%d')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("clusterer")
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
        "module": "clusterer",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))
# Constants
MIN_TITLES_FOR_CLUSTERING = 10
NUM_TOPICS = 5
GAP_THRESHOLD = 0.10

# Clustering backends
def _cluster_bertopic(titles: list[str]) -> list[dict]:
    import logging as _std_logging
    _std_logging.getLogger("sentence_transformers").setLevel(_std_logging.ERROR)
    _std_logging.getLogger("bertopic").setLevel(_std_logging.ERROR)

    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    n_topics = max(2, min(NUM_TOPICS, len(titles) // 4))

    topic_model = BERTopic(
        nr_topics=n_topics,
        min_topic_size=2,
        vectorizer_model=CountVectorizer(stop_words="english", min_df=1),
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(titles)
    clusters = []
    for topic_id in sorted(set(topics) - {-1}):
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        words = [w for w, _ in (topic_model.get_topic(topic_id) or [])[:3]]
        label = " + ".join(words) if words else f"theme_{topic_id}"
        clusters.append({
            "label": label,
            "size": len(indices),
            "titles": [titles[i] for i in indices[:3]],
        })
    outliers = [i for i, t in enumerate(topics) if t == -1]
    if outliers:
        clusters.append({
            "label": "general",
            "size": len(outliers),
            "titles": [titles[i] for i in outliers[:3]],
        })
    return clusters

def _cluster_tfidf_kmeans(titles: list[str]) -> list[dict]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_clusters = max(2, min(NUM_TOPICS, len(titles)))
    vec = TfidfVectorizer(stop_words="english", max_features=500, min_df=1)
    X = vec.fit_transform(titles)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    feature_names = vec.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    clusters = []
    for i in range(n_clusters):
        indices = [j for j, lbl in enumerate(labels) if lbl == i]
        top_words = [feature_names[w] for w in order_centroids[i, :3]]
        clusters.append({
            "label": " + ".join(top_words),
            "size": len(indices),
            "titles": [titles[j] for j in indices[:3]],
        })
    return clusters

# Per-keyword clustering
def _cluster_keyword(keyword: str, run_date: date, conn) -> list[dict]:
    df = fetch_df(conn, """
        SELECT title FROM etsy_listings
        WHERE keyword = %s AND run_date = %s
    """, (keyword, run_date))
    titles = df["title"].dropna().tolist()
    if len(titles) < MIN_TITLES_FOR_CLUSTERING:
        _log("INFO", f"Only {len(titles)} titles — using single general theme",
             keyword=keyword, run_date=str(run_date))
        return [{"label": "general", "size": len(titles),
                 "titles": titles[:3], "is_gap": False}]
    try:
        clusters = _cluster_bertopic(titles)
        method = "bertopic"
    except Exception as exc:
        _log("WARNING", f"BERTopic failed ({exc}) — falling back to TF-IDF/K-Means",
             keyword=keyword)
        clusters = _cluster_tfidf_kmeans(titles)
        method = "tfidf_kmeans"
    total = len(titles)
    for cluster in clusters:
        cluster["is_gap"] = (cluster["size"] / total) < GAP_THRESHOLD
    _log("INFO", f"Clustered into {len(clusters)} themes",
         keyword=keyword,
         method=method,
         run_date=str(run_date),
         themes=[c["label"] for c in clusters])
    return clusters

# DB write (idempotent — clears previous results for same date/keyword)
def _save_clusters(conn, run_date: date, keyword: str, clusters: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM niche_themes WHERE run_date = %s AND keyword = %s",
            (run_date, keyword),
        )
    conn.commit()
    with conn.cursor() as cur:
        for cluster in clusters:
            cur.execute("""
                INSERT INTO niche_themes
                    (run_date, keyword, theme_label, cluster_size,
                     representative_titles, is_gap)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                run_date,
                keyword,
                cluster["label"],
                cluster["size"],
                json.dumps(cluster["titles"]),
                int(cluster["is_gap"]),
            ))
    conn.commit()
    
# Main entry point
def compute_clusters(run_date: date) -> dict[str, list[dict]]:
    """Cluster Etsy listing titles for every viable keyword on run_date.
    Writes results to niche_themes. Returns {keyword: [cluster, ...]}."""
    conn = get_connection()
    viable_df = fetch_df(conn, """
        SELECT keyword FROM keyword_runs
        WHERE run_date = %s AND (is_viable IS NULL OR is_viable = 1)
    """, (run_date,))
    if viable_df.empty:
        _log("WARNING", "No viable keywords to cluster", run_date=str(run_date))
        conn.close()
        return {}
    results: dict[str, list[dict]] = {}
    for keyword in viable_df["keyword"].tolist():
        clusters = _cluster_keyword(keyword, run_date, conn)
        _save_clusters(conn, run_date, keyword, clusters)
        results[keyword] = clusters
    _log("INFO", f"Clustering complete for {len(results)} keywords",
         run_date=str(run_date))
    conn.close()
    return results


if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    results = compute_clusters(target)
    if not results:
        print("No results — run collector first for this date.")
    else:
        for keyword, clusters in results.items():
            print(f"\n{keyword}:")
            for c in clusters:
                gap = " [GAP]" if c["is_gap"] else ""
                print(f"  {c['label']:<45} {c['size']} listings{gap}")
