"""Pure, deterministic derivation of beliefs from events.

This is the single source of truth for how events become beliefs. It is used
both for the incremental refresh after every write and for a full
``rebuild()``; because it is a pure function of the ordered event list, the
derived view can always be thrown away and recomputed.

Rules (per subject, events processed in ``seq`` order):

* ``assert`` creates a belief whose valid time starts at ``valid_from``
  (explicit, or the event's ``recorded_at``). Its *predecessor* is the active
  belief immediately before it in valid time. The new belief ``supersedes``
  the predecessor and, if their polarities are opposite (+1 vs -1), also
  ``contradicts`` it. Relations are historical facts: they never change.
* ``forget`` soft-deletes a single belief (``target_id``) or every active
  belief of the subject. Forgotten beliefs stay in history, flagged.
* Valid-time intervals are re-chained over the *active* beliefs after each
  event: each belief is valid until the next active belief's ``valid_from``.
  A forgotten belief keeps the interval it had when it was forgotten (closed
  at the forget time if it was still open).
"""

from __future__ import annotations

from typing import Any, Iterable

Event = dict[str, Any]
BeliefRow = dict[str, Any]
Relation = tuple[str, str, str]  # (src_id, dst_id, kind)


def compute_timeline(events: Iterable[Event]) -> tuple[list[BeliefRow], list[Relation]]:
    beliefs: list[BeliefRow] = []
    by_id: dict[str, BeliefRow] = {}
    relations: list[Relation] = []

    def active() -> list[BeliefRow]:
        return [b for b in beliefs if not b["forgotten"]]

    def rechain() -> None:
        act = sorted(active(), key=lambda b: (b["valid_from"], b["seq"]))
        for i, b in enumerate(act):
            b["valid_to"] = act[i + 1]["valid_from"] if i + 1 < len(act) else None

    for ev in sorted(events, key=lambda e: e["seq"]):
        body = ev["body"]
        if ev["kind"] == "assert":
            valid_from = body.get("valid_from") or ev["recorded_at"]
            b: BeliefRow = {
                "id": ev["id"],
                "seq": ev["seq"],
                "subject": ev["subject"],
                "polarity": int(body.get("polarity", 0)),
                "value": body.get("value", ""),
                "text": body.get("text", ""),
                "source": body.get("source", ""),
                "confidence": float(body.get("confidence", 1.0)),
                "valid_from": valid_from,
                "valid_to": None,
                "recorded_at": ev["recorded_at"],
                "supersedes": None,
                "superseded_by": None,
                "contradicts": None,
                "forgotten": 0,
                "forgotten_at": None,
                "forgotten_reason": None,
                "metadata": body.get("metadata") or {},
            }
            key = (valid_from, ev["seq"])
            preds = [x for x in active() if (x["valid_from"], x["seq"]) <= key]
            if preds:
                p = max(preds, key=lambda x: (x["valid_from"], x["seq"]))
                b["supersedes"] = p["id"]
                p["superseded_by"] = b["id"]
                relations.append((b["id"], p["id"], "supersedes"))
                if p["polarity"] * b["polarity"] < 0:
                    b["contradicts"] = p["id"]
                    relations.append((b["id"], p["id"], "contradicts"))
            beliefs.append(b)
            by_id[b["id"]] = b
            rechain()
        elif ev["kind"] == "forget":
            target = body.get("target_id")
            if target:
                targets = [by_id[target]] if target in by_id else []
            else:
                targets = active()
            for t in targets:
                if t["forgotten"]:
                    continue
                if t["valid_to"] is None:
                    t["valid_to"] = max(ev["recorded_at"], t["valid_from"])
                t["forgotten"] = 1
                t["forgotten_at"] = ev["recorded_at"]
                t["forgotten_reason"] = body.get("reason")
            rechain()
        # unknown kinds are ignored for forward compatibility
    return beliefs, relations


def pick_valid(beliefs: Iterable[BeliefRow], at: str) -> BeliefRow | None:
    """The active belief whose valid-time interval contains ``at``."""
    best = None
    for b in beliefs:
        if b["forgotten"] or b["valid_from"] > at:
            continue
        if b["valid_to"] is not None and b["valid_to"] <= at:
            continue
        if best is None or (b["valid_from"], b["seq"]) > (best["valid_from"], best["seq"]):
            best = b
    return best
