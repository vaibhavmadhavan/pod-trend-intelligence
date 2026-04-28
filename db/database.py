import os
import time
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL is not set. Add it to your .env file.\n"
            "Format: postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres"
        )
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def run_migrations(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        conn.commit()

        cur.execute("SELECT version FROM schema_version")
        applied = {row[0] for row in cur.fetchall()}

    migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))

    for path in migration_files:
        version = int(path.stem.split("_")[0])
        if version in applied:
            continue

        sql = path.read_text(encoding="utf-8")
        statements = [s.strip() for s in sql.split(";\n") if s.strip()]

        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
            cur.execute(
                "INSERT INTO schema_version (version, description) VALUES (%s, %s)",
                (version, path.stem),
            )
        conn.commit()


def retry_with_backoff(fn, max_attempts=3):
    delays = [2, 4, 8]
    last_exc = None
    for i in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i < max_attempts - 1:
                time.sleep(delays[i])
    raise last_exc


def insert_or_ignore(conn, table, row):
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["%s"] * len(row))
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            list(row.values()),
        )
    conn.commit()


def update_row(conn, table, updates, where):
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    where_clause = " AND ".join(f"{k} = %s" for k in where)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET {set_clause} WHERE {where_clause}",
            [*updates.values(), *where.values()],
        )
    conn.commit()


def fetch_df(conn, query, params=None):
    with conn.cursor() as cur:
        cur.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    conn = get_connection()
    run_migrations(conn)
    df = fetch_df(
        conn,
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name",
    )
    print(df["table_name"].tolist())
    conn.close()
