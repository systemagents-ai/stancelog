"""Timestamp helpers.

All timestamps are stored as fixed-width ISO-8601 UTC strings
(``YYYY-MM-DDTHH:MM:SS.ffffffZ``) so that lexicographic order == time order,
which lets SQLite compare them directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

TimeLike = Union[datetime, str, int, float]

_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def now() -> str:
    return fmt(datetime.now(timezone.utc))


def fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        # Naive datetimes are interpreted as UTC (documented behaviour).
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(_FMT)


def parse(value: TimeLike) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise TypeError("timestamp must not be a bool")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    raise TypeError(f"unsupported timestamp type: {type(value).__name__}")


def normalize(value: TimeLike | None) -> str | None:
    """Convert any accepted time representation into the canonical string."""
    if value is None:
        return None
    return fmt(parse(value))


def age_seconds(ts: str, reference: str | None = None) -> float:
    ref = parse(reference) if reference else datetime.now(timezone.utc)
    return max(0.0, (ref - parse(ts)).total_seconds())
