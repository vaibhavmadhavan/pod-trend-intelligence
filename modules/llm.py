import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from db.database import fetch_df, get_connection

load_dotenv()

# Logging
def _setup_logger() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    log_path = config.LOG_DIR / f"pipeline_{date.today().strftime('%Y%m%d')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("llm")
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
        "module": "llm",
        "message": message,
        **extra,
    }
    getattr(_logger, level.lower())(json.dumps(entry))

#Output dataclass
@dataclass
class LLMOutput:
    keyword: str
    run_date: date
    strategic_insight: str
    recommended_angle: str
    slogans: list[str]
    etsy_title: str
    etsy_tags: list[str]
    model_used: str
    prompt_version: str
_jinja_env = Environment(
    loader=FileSystemLoader(str(config.TEMPLATE_DIR)),
    autoescape=False,
)
def _render_prompt(keyword: str, context: dict) -> str:
    template = _jinja_env.get_template("llm_prompt_v1.j2")
    return template.render(keyword=keyword, **context)

# JSON parsing — strips markdown fences if the model adds them
def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)

# Provider calls

def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=config.LLM_MODEL_OPENAI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content
def _call_ollama(prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model=config.LLM_MODEL_OLLAMA,
        messages=[{"role": "user", "content": prompt}],
    )
    if hasattr(response, "message"):
        return response.message.content
    return response["message"]["content"]

# Core generation
def generate_copy(keyword: str, context: dict, run_date: date) -> LLMOutput | None:
    prompt = _render_prompt(keyword, context)
    raw_text = None
    # Try OpenAI first if key is present
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            raw_text = _call_openai(prompt)
            model_used = config.LLM_MODEL_OPENAI
            _log("INFO", "OpenAI call succeeded", keyword=keyword)
        except Exception as exc:
            _log("WARNING", f"OpenAI failed ({exc}) — trying Ollama", keyword=keyword)
    # Fallback to Ollama
    if raw_text is None:
        try:
            raw_text = _call_ollama(prompt)
            model_used = config.LLM_MODEL_OLLAMA
            _log("INFO", "Ollama call succeeded", keyword=keyword)
        except Exception as exc:
            _log("ERROR", f"Ollama failed ({exc}) — skipping keyword", keyword=keyword)
            return None
    try:
        data = _parse_json(raw_text)
    except Exception as exc:
        _log("ERROR", f"JSON parse failed ({exc})", keyword=keyword, raw=raw_text[:300])
        return None
    return LLMOutput(
        keyword=keyword,
        run_date=run_date,
        strategic_insight=data.get("strategic_insight", ""),
        recommended_angle=data.get("recommended_angle", ""),
        slogans=data.get("slogans", []),
        etsy_title=data.get("etsy_title", ""),
        etsy_tags=data.get("etsy_tags", []),
        model_used=model_used,
        prompt_version=config.LLM_PROMPT_VERSION,
    )

# DB helpers
def _already_processed(conn, run_date: date, keyword: str) -> bool:
    df = fetch_df(
        conn,
        "SELECT 1 FROM llm_outputs WHERE run_date = %s AND keyword = %s",
        (run_date, keyword),
    )
    return not df.empty
def _save_output(conn, output: LLMOutput) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO llm_outputs
                (run_date, keyword, strategic_insight, recommended_angle,
                 slogans, etsy_title, etsy_tags, model_used, prompt_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_date, keyword) DO NOTHING
        """, (
            output.run_date,
            output.keyword,
            output.strategic_insight,
            output.recommended_angle,
            json.dumps(output.slogans),
            output.etsy_title,
            json.dumps(output.etsy_tags),
            output.model_used,
            output.prompt_version,
        ))
    conn.commit()
# Main entry point
def compute_llm_outputs(run_date: date) -> list[LLMOutput]:
    """Generate Etsy copy for every viable keyword on run_date.
    Skips keywords already processed today (idempotent).
    Returns list of LLMOutput objects."""
    conn = get_connection()
    viable_df = fetch_df(conn, """
        SELECT keyword, opportunity_score, trend_direction,
               etsy_demand, competition_score
        FROM keyword_runs
        WHERE run_date = %s AND (is_viable IS NULL OR is_viable = 1)
    """, (run_date,))
    if viable_df.empty:
        _log("WARNING", "No viable keywords found", run_date=str(run_date))
        conn.close()
        return []
    results: list[LLMOutput] = []
    for _, row in viable_df.iterrows():
        keyword = row["keyword"]
        if _already_processed(conn, run_date, keyword):
            _log("INFO", "Already processed today — skipping", keyword=keyword)
            continue
        themes_df = fetch_df(conn, """
            SELECT theme_label AS label, cluster_size AS size, is_gap
            FROM niche_themes
            WHERE run_date = %s AND keyword = %s
            ORDER BY cluster_size DESC
        """
        , (run_date, keyword))
        themes = themes_df.to_dict("records") if not themes_df.empty else []
        context = {
            "opportunity_score": round(float(row["opportunity_score"] or 0), 1),
            "trend_direction": row["trend_direction"] or "Unknown",
            "etsy_demand": round(float(row["etsy_demand"] or 0), 1),
            "competition_score": round(float(row["competition_score"] or 0), 3),
            "themes": themes,
        }
        output = generate_copy(keyword, context, run_date)
        if output is None:
            continue
        _save_output(conn, output)
        results.append(output)
        _log("INFO", "Copy generated and saved",
             keyword=keyword,
             model=output.model_used,
             etsy_title=output.etsy_title,
             run_date=str(run_date))
    _log("INFO", f"LLM outputs complete: {len(results)} generated",
         run_date=str(run_date))
    conn.close()
    return results

if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    outputs = compute_llm_outputs(target)
    if not outputs:
        print("No outputs — run collector + clusterer first, or check Ollama is running.")
    else:
        for o in outputs:
            print(f"\n{'='*60}")
            print(f"Keyword:   {o.keyword}")
            print(f"Model:     {o.model_used}")
            print(f"Title:     {o.etsy_title}")
            print(f"Angle:     {o.recommended_angle}")
            print(f"Insight:   {o.strategic_insight}")
            print(f"Slogans:   {', '.join(o.slogans)}")
            print(f"Tags:      {', '.join(o.etsy_tags)}")
