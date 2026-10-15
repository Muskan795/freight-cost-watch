"""Deterministic pipeline: shipments + reviewed note facts -> output.csv.

  python run.py [--out-dir DIR]
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

import config as cfg
from costs import add_baselines, add_flags, build_weekly, load_shipments
from notes import Retriever, explain, load_notes

DEFAULT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                   "vs_similar_routes", "flagged", "matched_note_id", "reason"]


def output_columns():
    """The sample file's header is the contract: use its exact names and order."""
    if not Path(cfg.SAMPLE_FILE).exists():
        return DEFAULT_COLUMNS
    with open(cfg.SAMPLE_FILE, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    if set(header) != set(DEFAULT_COLUMNS):
        raise ValueError(f"Sample header {header} differs from expected {DEFAULT_COLUMNS}; "
                         "update DEFAULT_COLUMNS and the output mapping.")
    return header


def analyse():
    shipments, quality = load_shipments(cfg.SHIPMENTS_FILE)
    weekly = add_flags(add_baselines(build_weekly(shipments)))

    notes = load_notes(pd.read_csv(cfg.NOTES_FILE), cfg.NOTE_FACTS_FILE)
    retriever = Retriever(notes)

    ids, reasons, audits = [], [], []
    for _, row in weekly.iterrows():
        if not row["flagged"]:
            ids.append(""), reasons.append("")
            continue
        note_id, reason, audit = explain(row, notes, retriever)
        ids.append(note_id), reasons.append(reason), audits.extend(audit)
    weekly["matched_note_id"], weekly["reason"] = ids, reasons
    return weekly, audits, quality, notes


def to_output(weekly, columns):
    out = weekly.copy()
    out["week_of"] = out["week_of"].dt.strftime("%Y-%m-%d")
    for col in ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes"]:
        out[col] = out[col].round(cfg.DECIMALS)
    out["flagged"] = out["flagged"].map({True: cfg.FLAGGED_TRUE, False: cfg.FLAGGED_FALSE})
    return out[columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weekly, audits, quality, notes = analyse()
    columns = output_columns()
    output = to_output(weekly, columns)
    output.to_csv(out_dir / cfg.OUTPUT_FILE, index=False, na_rep="")
    pd.DataFrame(audits).to_csv(out_dir / cfg.AUDIT_FILE, index=False)

    # ---- sanity checks on what was written ----
    written = pd.read_csv(out_dir / cfg.OUTPUT_FILE, keep_default_na=False)
    assert written.columns.tolist() == columns
    assert len(written) == len(weekly)
    assert not written.duplicated(["route", "week_of"]).any()
    justified = written["reason"].str.contains("Justified by note")
    assert (justified == (written["matched_note_id"] != "")).all(), "verdict/note-id mismatch"

    unreviewed = [n.note_id for n in notes.values() if n.needs_review]
    print("Data quality:", quality)
    print(f"Route-weeks: {len(written)}  flagged: {int(weekly['flagged'].sum())}  "
          f"justified: {int(justified.sum())}")
    if unreviewed:
        print("WARNING: notes with unverified facts (treated as unusable):", unreviewed)
    print(f"Wrote {out_dir / cfg.OUTPUT_FILE}")


if __name__ == "__main__":
    main()
