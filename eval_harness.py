"""Checks that justify trusting the output.

  python eval_harness.py            # unit tests, plus labelled-set scoring if labels.csv exists

labels.csv (you create ~20 rows by hand from the raw notes, WITHOUT looking at output.csv):
  route,week_of,expected_verdict,expected_note_id      (verdict: justified | unexplained)
"""
from pathlib import Path

import pandas as pd

import config as cfg
from costs import add_baselines, add_flags, build_weekly
from notes import Note, check_note

D = pd.Timestamp


def ships(rows):
    """rows: (origin, dest, type, date, cost). qty=1 t, 1 km -> cost == cost/tonne-km."""
    return pd.DataFrame([{"shipment_id": i, "origin": o, "destination": d, "route_type": t,
                          "quantity_tonnes": 1.0, "distance_km": 1.0, "freight_cost_inr": c,
                          "shipment_date": pd.Timestamp(dt)} for i, (o, d, t, dt, c) in enumerate(rows)])


def test_pooling_not_average_of_ratios():
    df = ships([("A", "B", "Short", "2025-01-06", 100)])
    df = pd.concat([df, pd.DataFrame([{"shipment_id": 9, "origin": "A", "destination": "B",
        "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 20.0,
        "freight_cost_inr": 600.0, "shipment_date": D("2025-01-07")}])], ignore_index=True)
    df.loc[0, ["quantity_tonnes", "distance_km"]] = [10.0, 10.0]      # ratio 1.0; other is 3.0
    w = build_weekly(df)
    assert abs(w.loc[0, "cost_per_tonne_km"] - 700 / 300) < 1e-9      # not (1+3)/2


def test_weeks_run_monday_to_sunday():
    w = build_weekly(ships([("A", "B", "Short", "2025-01-12", 1),    # Sunday
                            ("A", "B", "Short", "2025-01-13", 1)]))  # next Monday
    assert [f"{d:%Y-%m-%d}" for d in w["week_of"]] == ["2025-01-06", "2025-01-13"]


def test_own_history_window_has_no_lookahead():
    rows = [("A", "B", "Short", D("2025-01-06") + pd.Timedelta(weeks=i), float(i + 1)) for i in range(10)]
    w = add_baselines(build_weekly(ships(rows)))
    assert pd.isna(w.loc[0, "own_history_avg"])                       # no prior weeks
    assert w.loc[2, "own_history_avg"] == 1.5                         # uses all prior (1,2)
    assert w.loc[9, "own_history_avg"] == 5.5                         # weeks 2..9 only, not 1 or 10
    assert w.loc[2, "own_history_weeks"] == 2


def test_peer_average_excludes_self():
    rows = [("A", "B", "Short", "2025-01-06", 1), ("C", "D", "Short", "2025-01-06", 2),
            ("E", "F", "Short", "2025-01-06", 6), ("G", "H", "Long", "2025-01-06", 50)]
    w = add_baselines(build_weekly(ships(rows))).set_index("route")
    assert w.loc["A-B", "peer_avg"] == 4.0 and w.loc["E-F", "peer_avg"] == 1.5
    assert pd.isna(w.loc["G-H", "peer_avg"])                          # no Long peers


def test_threshold_flagging():
    rows = [("A", "B", "Short", D("2025-01-06") + pd.Timedelta(weeks=i), c)
            for i, c in enumerate([10, 10, 10, 11.0, 10.9])]
    w = add_flags(add_baselines(build_weekly(ships(rows))))
    assert w["flagged"].tolist() == [False, False, False, True, False]  # 10% yes, 9% no


def _note(**kw):
    base = dict(note_id="N", note_date=D("2025-01-01"), applies_to="A-B", text="t", in_dataset=True,
                effect="increase", start=D("2025-02-03"), end=D("2025-02-09"), basis="explicit",
                needs_review=False)
    return Note(**{**base, **kw})


def test_note_gate_rejects_bad_matches():
    wk = D("2025-02-03")
    assert check_note(_note(), "A-B", "Short", wk)[0] == 2                          # valid
    assert check_note(_note(applies_to="All Routes"), "A-B", "Short", wk)[0] == 0   # nationwide ok
    assert check_note(_note(applies_to="C-D"), "A-B", "Short", wk)[0] is None       # wrong route
    assert check_note(_note(), "A-B", "Short", D("2025-03-03"))[0] is None          # wrong window
    assert check_note(_note(effect="none"), "A-B", "Short", wk)[0] is None          # no cost effect
    assert check_note(_note(effect="decrease"), "A-B", "Short", wk)[0] is None      # wrong direction
    assert check_note(_note(in_dataset=False), "A-B", "Short", wk)[0] is None       # out of dataset
    assert check_note(_note(needs_review=True), "A-B", "Short", wk)[0] is None      # unverified facts
    assert check_note(_note(end=None), "A-B", "Short", wk)[0] is None               # no window
    assert check_note(_note(), "A-B", "Short", D("2025-02-10"))[0] is None          # week after window


def score_labels(path="labels.csv"):
    from run import analyse
    weekly, _, _, _ = analyse()
    weekly["week_of"] = weekly["week_of"].dt.strftime("%Y-%m-%d")
    labels = pd.read_csv(path, keep_default_na=False)
    merged = labels.merge(weekly, on=["route", "week_of"], how="left")
    verdict = merged["reason"].str.contains("Justified by note").map({True: "justified", False: "unexplained"})
    verdict_ok = verdict == merged["expected_verdict"]
    note_ok = merged["matched_note_id"] == merged["expected_note_id"]
    print(f"\nLabelled set: {len(labels)} rows | verdict accuracy {verdict_ok.mean():.0%} "
          f"| note-id accuracy {note_ok.mean():.0%}")
    for _, r in merged[~(verdict_ok & note_ok)].iterrows():
        print(f"  MISMATCH {r['route']} {r['week_of']}: expected {r['expected_verdict']}/"
              f"{r['expected_note_id'] or '-'}, got note {r['matched_note_id'] or '-'}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} unit tests passed")
    if Path("labels.csv").exists():
        score_labels()
    else:
        print("No labels.csv found - create one to score against a hand-labelled set.")
