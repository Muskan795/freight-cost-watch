"""Context-note retrieval and verification.

Pipeline for each flagged route-week:
  1. RETRIEVE  - rank all notes against the route/week (TF-IDF vectors, deterministic).
  2. VERIFY    - accept a note only if ALL hold: facts reviewed, concerns this dataset,
                 applies to this route / route_type / all routes, says costs went UP,
                 and its dated window overlaps the week.
                 A note that covers all routes or a whole route type also hits the peer
                 routes, so it may only explain a rise versus the route's OWN history,
                 never a gap versus similar routes.
  3. EXPLAIN   - the reason quotes the note verbatim; no free text is generated.
No accepted note -> "Unexplained". The LLM never decides a verdict.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg

NOTE_COLUMNS = ["note_id", "date", "applies_to", "note"]


@dataclass(frozen=True)
class Note:
    note_id: str
    note_date: pd.Timestamp
    applies_to: str
    text: str
    in_dataset: bool = True
    effect: str = "unknown"            # increase | decrease | none | unknown
    start: pd.Timestamp = None
    end: pd.Timestamp = None
    basis: str = ""                    # explicit | inferred
    needs_review: bool = True


def norm_scope(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _ts(value):
    return pd.Timestamp(value) if value else None


def load_notes(notes_df, facts_path):
    """Join the raw notes with their reviewed structured facts."""
    missing = set(NOTE_COLUMNS) - set(notes_df.columns)
    if missing:
        raise ValueError(f"Missing context-note columns: {sorted(missing)}")
    facts = json.loads(Path(facts_path).read_text(encoding="utf-8"))

    notes = {}
    for r in notes_df.itertuples():
        f = facts.get(r.note_id)
        if f is None:
            notes[r.note_id] = Note(r.note_id, pd.Timestamp(r.date), r.applies_to, r.note)
            continue
        start, end = _ts(f.get("start")), _ts(f.get("end"))
        if end is None and start is not None and f.get("open_ended"):
            end = start + pd.Timedelta(weeks=cfg.MAX_PERSISTENT_WEEKS) - pd.Timedelta(days=1)
        notes[r.note_id] = Note(
            r.note_id, pd.Timestamp(r.date), r.applies_to, r.note,
            in_dataset=bool(f.get("in_dataset", True)),
            effect=f.get("effect", "unknown"),
            start=start, end=end,
            basis=f.get("date_basis", ""),
            needs_review=bool(f.get("needs_review", False)),
        )
    return notes


class Retriever:
    """Vector retrieval over the notes (TF-IDF: free, offline, fully deterministic).

    Swap in sentence-transformer embeddings + Chroma/FAISS by re-implementing top_k();
    the verification gate below is what guarantees correctness either way.
    """

    def __init__(self, notes):
        self.ids = sorted(notes)
        docs = [f"{notes[i].applies_to} {notes[i].text}" for i in self.ids]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(docs)

    def top_k(self, query, k):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sorted(range(len(self.ids)), key=lambda i: (-round(float(sims[i]), 10), self.ids[i]))
        return [(self.ids[i], float(sims[i])) for i in order[:k]]


def check_note(note, route, route_type, week_start):
    """Return (scope_rank, '') if the note may justify this week, else (None, why not)."""
    if note.needs_review:
        return None, "structured facts missing or not verified"
    if not note.in_dataset:
        return None, "note concerns routes outside this dataset"

    scope = norm_scope(note.applies_to)
    if scope == norm_scope(route):
        rank = 2
    elif scope == norm_scope(route_type):
        rank = 1
    elif scope in ("all-routes", "all"):
        rank = 0
    else:
        return None, "applies to a different route"

    if note.effect != "increase":
        return None, f"note effect is '{note.effect}', not a cost increase"
    if note.start is None or note.end is None:
        return None, "note has no supported date window"
    if not (week_start <= note.end and week_start + pd.Timedelta(days=6) >= note.start):
        return None, "note window does not overlap this week"
    return rank, ""


def explain(row, notes, retriever):
    """Return (matched_note_id, reason, audit_rows) for one flagged route-week."""
    route, week = row["route"], row["week_of"]
    query = f"{route} {row['route_type']} freight cost increase {week:%B %Y}"

    # True only if the route rose by the threshold versus its OWN history (NaN -> False).
    own_history_hit = bool(row["vs_own_history"] >= cfg.ANOMALY_THRESHOLD_PCT)

    accepted, audit = [], []
    for note_id, score in retriever.top_k(query, cfg.RETRIEVAL_TOP_K):
        rank, why = check_note(notes[note_id], route, row["route_type"], week)

        # A note covering all routes / a whole route type raises the peers' costs too,
        # so it cannot explain a route being dearer than its peers. It may only explain
        # a rise versus the route's own history.
        if rank is not None and rank < 2 and not own_history_hit:
            rank, why = None, "wide-scope note cannot explain a gap versus similar routes"

        audit.append({"route": route, "week_of": f"{week:%Y-%m-%d}", "note_id": note_id,
                      "retrieval_score": round(score, 4),
                      "accepted": rank is not None, "reason": why or "passes all checks"})
        if rank is not None:
            accepted.append((rank, notes[note_id].note_date, note_id))

    if not accepted:
        return "", (f"{row['anomaly_text']}. Unexplained: no context note applies to this "
                    "route and week with an upward cost effect."), audit

    # Most specific scope first, then latest note date, then id (fully deterministic).
    accepted.sort(key=lambda a: (-a[0], -a[1].value, a[2]))
    note = notes[accepted[0][2]]
    window = f"{note.start:%d %b %Y} to {note.end:%d %b %Y}"
    basis = " (end date inferred from the note's wording)" if note.basis == "inferred" else ""
    reason = (f"{row['anomaly_text']}. Justified by note {note.note_id} "
              f"(dated {note.note_date:%d %b %Y}, applies to {note.applies_to}): "
              f"\"{note.text}\" Its window, {window}{basis}, overlaps the week of {week:%d %b %Y}.")
    return note.note_id, reason, audit