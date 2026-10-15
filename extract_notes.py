"""ONE-TIME step: use an LLM to turn each free-text note into structured facts.

  Put GROQ_API_KEY=... in a .env file next to this script (never commit it)
  python extract_notes.py          -> note_facts.json + token_log.json
  python extract_notes.py --force  -> re-extract notes already cached

Design: one call per NOTE (not per flagged row), temperature 0, results cached.
run.py never calls an LLM, so the pipeline's output is identical on every run.
Review note_facts.json by hand before running run.py; the LLM output is a draft.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config as cfg

SYSTEM_PROMPT = """You extract structured facts from short freight-cost notes.
Use ONLY the note text and its note date. Never guess. Reply with one JSON object:
{
 "in_dataset": false if the note says the affected routes/places are not part of the dataset, else true,
 "effect": "increase" if the note says costs rose / a surcharge / a disruption raised cost;
           "decrease" if costs fell; "none" if the note says costs were not affected,
           returned to normal, or is unrelated to cost; "unknown" if unclear,
 "start": "YYYY-MM-DD" or null   (only if the text states or clearly implies a start),
 "end": "YYYY-MM-DD" or null     (only if stated; or start+6 days if text says 'about a week' / 'one week'),
 "date_basis": "explicit" if both dates are written in the text, "inferred" if you derived the end,
 "open_ended": true only if the note says the change continues/is permanent from the start date,
 "evidence": "verbatim quote (max 25 words) from the note supporting effect and dates"
}
Use the note date to infer the year when the text gives only month and day."""

MAX_TRIES = 3


def call_llm(client, row):
    """One extraction call. Returns (fact_dict, usage). Bad JSON -> empty fact (flagged by validate)."""
    user = (f"note_id: {row.note_id}\nnote_date: {row.date}\n"
            f"applies_to: {row.applies_to}\nnote: {row.note}")
    resp = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        temperature=cfg.LLM_TEMPERATURE,
        seed=cfg.LLM_SEED,
        reasoning_effort="low",          # gpt-oss is a reasoning model; keep thinking short
        max_completion_tokens=2000,      # room for reasoning tokens AND the JSON answer
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    try:
        fact = json.loads(resp.choices[0].message.content)
        if not isinstance(fact, dict):
            raise ValueError("not a JSON object")
    except (TypeError, ValueError):       # JSONDecodeError is a ValueError
        fact = {"effect": "unknown", "evidence": ""}   # validate() marks it needs_review
    return fact, resp.usage


def call_with_retry(client, row):
    """Retry a few times on API errors (rate limit, timeout). Raises after the last try."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return call_llm(client, row)
        except Exception as exc:          # groq raises several error types
            if attempt == MAX_TRIES:
                raise
            wait = 2 ** attempt
            print(f"  {row.note_id}: {type(exc).__name__}, retrying in {wait}s")
            time.sleep(wait)


def validate(fact, note_text):
    """Mark the fact for manual review if it is malformed or its evidence is not verbatim."""
    ok = fact.get("effect") in {"increase", "decrease", "none", "unknown"}
    for key in ("start", "end"):
        if fact.get(key):
            try:
                pd.Timestamp(fact[key])
            except (ValueError, TypeError):
                ok = False
    if ok and fact.get("start") and fact.get("end") \
            and pd.Timestamp(fact["end"]) < pd.Timestamp(fact["start"]):
        ok = False
    evidence = " ".join(str(fact.get("evidence", "")).lower().split())
    if not evidence or evidence not in " ".join(str(note_text).lower().split()):
        ok = False
    fact["needs_review"] = not ok
    return fact


def save(facts, log):
    """Write both files. Called after every note so a crash never loses finished work."""
    tin = sum(c["input_tokens"] for c in log["calls"])
    tout = sum(c["output_tokens"] for c in log["calls"])
    log.update({
        "model": cfg.LLM_MODEL,
        "temperature": cfg.LLM_TEMPERATURE,
        "num_calls": len(log["calls"]),
        "total_input_tokens": tin,
        "total_output_tokens": tout,   # includes the model's hidden reasoning tokens
        "estimated_cost_usd": round(tin / 1e6 * cfg.PRICE_IN_PER_M_USD
                                    + tout / 1e6 * cfg.PRICE_OUT_PER_M_USD, 6),
    })
    Path(cfg.NOTE_FACTS_FILE).write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
    Path(cfg.TOKEN_LOG_FILE).write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def main():
    load_dotenv(override=True)            # .env wins over any old `export` in the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-extract notes already cached")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not found. Put it in a .env file next to this script.")

    from groq import Groq                 # pip install groq
    client = Groq(api_key=api_key)

    notes = pd.read_csv(cfg.NOTES_FILE)
    facts_path, log_path = Path(cfg.NOTE_FACTS_FILE), Path(cfg.TOKEN_LOG_FILE)
    facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"calls": []}

    failed = []
    for row in notes.itertuples():
        if row.note_id in facts and not args.force:
            continue
        try:
            fact, usage = call_with_retry(client, row)
        except Exception as exc:
            print(f"{row.note_id}: FAILED ({type(exc).__name__}: {exc})")
            failed.append(row.note_id)
            continue
        facts[row.note_id] = validate(fact, row.note)
        log["calls"].append({"note_id": row.note_id,
                             "input_tokens": usage.prompt_tokens,
                             "output_tokens": usage.completion_tokens})
        print(f"{row.note_id}: effect={fact.get('effect')} needs_review={fact['needs_review']}")
        save(facts, log)

    log = save(facts, log)
    print(f"\nCalls: {log['num_calls']}  in: {log['total_input_tokens']}  "
          f"out: {log['total_output_tokens']}  cost: ${log['estimated_cost_usd']}")
    if failed:
        print("Not extracted (run again to retry):", failed)
    flagged = [k for k, v in facts.items() if v.get("needs_review")]
    if flagged:
        print("Review these by hand in note_facts.json:", sorted(flagged))


if __name__ == "__main__":
    main()