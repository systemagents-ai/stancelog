"""Event hashing (tamper/corruption evidence for the append-only log)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def event_hash(ev: dict[str, Any]) -> str:
    """sha256 over the previous hash and every stored field of the event.

    Chaining ``prev_hash`` means a missing, reordered or modified event breaks
    verification of everything after it.
    """
    parts = [
        ev["prev_hash"],
        str(ev["seq"]),
        ev["id"],
        ev["kind"],
        ev["subject"] if ev["subject"] is not None else "",
        ev["recorded_at"],
        ev["body_json"],
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
