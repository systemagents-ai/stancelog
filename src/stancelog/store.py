"""The :class:`Memory` store: SQLite event log + derived belief view + JSONL journal."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from . import _time
from .extract import Claim, Extractor, HeuristicExtractor, LLMExtractor, normalize_subject
from .hashing import GENESIS, canonical_json, event_hash
from .journal import Journal, event_to_record
from .search import bm25_scores, query_terms, tokenize
from .timeline import compute_timeline, pick_valid

__all__ = ["Memory", "Belief", "RecoveryError", "default_db_path"]

SCHEMA_VERSION = "1"
ENV_DB = "STANCELOG_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The source of truth. Append-only: enforced by triggers below.
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY,
    id          TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    subject     TEXT,
    body        TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_subject ON events(subject, seq);

CREATE TRIGGER IF NOT EXISTS events_append_only_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'stancelog: events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_append_only_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'stancelog: events are append-only'); END;

-- Derived view (rebuildable from events at any time).
CREATE TABLE IF NOT EXISTS beliefs (
    id               TEXT PRIMARY KEY,
    seq              INTEGER NOT NULL,
    subject          TEXT NOT NULL,
    polarity         INTEGER NOT NULL,
    value            TEXT NOT NULL,
    text             TEXT NOT NULL,
    source           TEXT NOT NULL,
    confidence       REAL NOT NULL,
    valid_from       TEXT NOT NULL,
    valid_to         TEXT,
    recorded_at      TEXT NOT NULL,
    supersedes       TEXT,
    superseded_by    TEXT,
    contradicts      TEXT,
    forgotten        INTEGER NOT NULL DEFAULT 0,
    forgotten_at     TEXT,
    forgotten_reason TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS beliefs_by_subject ON beliefs(subject, valid_from, seq);
CREATE INDEX IF NOT EXISTS beliefs_by_validity ON beliefs(forgotten, valid_to);

CREATE TABLE IF NOT EXISTS relations (
    src     TEXT NOT NULL,
    dst     TEXT NOT NULL,
    kind    TEXT NOT NULL,
    subject TEXT NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS relations_by_subject ON relations(subject);
"""

_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts USING fts5("
    "subject, body, tokenize='porter unicode61')"
)

_BELIEF_COLS = (
    "id", "seq", "subject", "polarity", "value", "text", "source", "confidence",
    "valid_from", "valid_to", "recorded_at", "supersedes", "superseded_by", "contradicts",
    "forgotten", "forgotten_at", "forgotten_reason", "metadata",
)


class RecoveryError(RuntimeError):
    """Raised when the store cannot be opened or recovered."""


class _NeedsRecovery(Exception):
    pass


def default_db_path() -> str:
    env = os.environ.get(ENV_DB)
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".stancelog", "memory.db")


@dataclass
class Belief:
    """A versioned stance on a subject, as derived from the event log."""

    id: str
    subject: str
    polarity: int
    value: str
    text: str
    source: str
    confidence: float
    valid_from: str
    valid_to: str | None
    recorded_at: str
    seq: int
    supersedes: str | None = None
    superseded_by: str | None = None
    contradicts: str | None = None
    forgotten: bool = False
    forgotten_at: str | None = None
    forgotten_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    @property
    def stance(self) -> str:
        return {1: "positive", -1: "negative", 0: "neutral"}[self.polarity]

    @property
    def is_open(self) -> bool:
        """True if not forgotten and its valid-time interval is still open."""
        return not self.forgotten and self.valid_to is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stance"] = self.stance
        if d["score"] is None:
            d.pop("score")
        return d

    @classmethod
    def _from_row(cls, row: sqlite3.Row | dict[str, Any], score: float | None = None) -> "Belief":
        r = dict(row)
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        return cls(
            id=r["id"], subject=r["subject"], polarity=int(r["polarity"]), value=r["value"],
            text=r["text"], source=r["source"], confidence=float(r["confidence"]),
            valid_from=r["valid_from"], valid_to=r["valid_to"], recorded_at=r["recorded_at"],
            seq=int(r["seq"]), supersedes=r["supersedes"], superseded_by=r["superseded_by"],
            contradicts=r["contradicts"], forgotten=bool(r["forgotten"]),
            forgotten_at=r["forgotten_at"], forgotten_reason=r["forgotten_reason"],
            metadata=meta, score=score,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _is_corruption(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None and (code & 0xFF) in (11, 26):  # SQLITE_CORRUPT, SQLITE_NOTADB
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("malformed", "not a database", "corrupt", "file is encrypted"))


def _is_busy(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "seq": row["seq"], "id": row["id"], "kind": row["kind"], "subject": row["subject"],
        "body_json": row["body"], "recorded_at": row["recorded_at"],
        "prev_hash": row["prev_hash"], "hash": row["hash"],
    }


def _with_body(ev: dict[str, Any]) -> dict[str, Any]:
    out = dict(ev)
    try:
        out["body"] = json.loads(ev["body_json"])
    except (json.JSONDecodeError, TypeError):
        out["body"] = {}
    return out


class _FileLock:
    """Cross-process exclusive lock (flock on POSIX, msvcrt on Windows, no-op otherwise)."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "_FileLock":
        if not self.path:
            return self
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows
            try:
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(self.fd, msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            except ImportError:
                pass
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.fd is None:
            return
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows
            try:
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            except (ImportError, OSError):
                pass
        os.close(self.fd)
        self.fd = None


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

class Memory:
    """Durable, contradiction-aware agent memory.

    Args:
        path: SQLite file. Defaults to ``$STANCELOG_DB`` or ``~/.stancelog/memory.db``.
            ``":memory:"`` gives a throwaway in-process store (no journal).
        extractor: anything implementing :class:`~stancelog.Extractor`;
            defaults to :class:`~stancelog.HeuristicExtractor`.
        journal: keep the fsync'd JSONL mirror ``<path>.journal.jsonl`` used
            for corruption recovery (default True).
        check_on_open: run ``PRAGMA quick_check`` on open and auto-recover.
        busy_timeout: seconds to wait for other writers before giving up.
        use_fts: force FTS5 on/off for :meth:`recall` (default: auto-detect).
        half_life_days / recency_weight: recall ranking knobs.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        extractor: Extractor | None = None,
        journal: bool = True,
        check_on_open: bool = True,
        busy_timeout: float = 30.0,
        use_fts: bool | None = None,
        half_life_days: float = 30.0,
        recency_weight: float = 0.3,
    ) -> None:
        raw = os.fspath(path) if path is not None else default_db_path()
        self._in_memory = raw == ":memory:" or raw.startswith("file::memory:")
        self.path = raw if self._in_memory else os.path.abspath(os.path.expanduser(raw))
        if not self._in_memory:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.journal_path = None if self._in_memory or not journal else self.path + ".journal.jsonl"
        self._journal = Journal(self.journal_path) if self.journal_path else None
        self._lock_path = None if self._in_memory else self.path + ".lock"
        self.extractor: Extractor = extractor or HeuristicExtractor()
        if isinstance(self.extractor, LLMExtractor) and self.extractor.known_subjects is None:
            self.extractor.known_subjects = lambda: self.subjects(limit=200)
        self.busy_timeout = float(busy_timeout)
        self.half_life_days = float(half_life_days)
        self.recency_weight = float(recency_weight)
        self._want_fts = use_fts
        self._fts = False
        self._lock = threading.RLock()
        self._tx_journaled: list[str] = []
        self._conn: sqlite3.Connection | None = None
        self.recovery: dict[str, Any] | None = None
        self._check_on_open = check_on_open
        self._open()

    # ------------------------------------------------------------------ open
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=self.busy_timeout, isolation_level=None, check_same_thread=False,
            uri=self.path.startswith("file:"),
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)}")
        if not self._in_memory:
            deadline = time.monotonic() + self.busy_timeout
            while True:
                try:
                    conn.execute("PRAGMA journal_mode=WAL").fetchone()
                    break
                except sqlite3.OperationalError as e:
                    if not _is_busy(e) or time.monotonic() > deadline:
                        conn.close()
                        raise
                    time.sleep(0.02)
        # FULL: in WAL mode every COMMIT fsyncs the WAL, so a committed write
        # survives power loss / OS crash, not just process death.
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _open(self) -> None:
        with _FileLock(self._lock_path):
            try:
                self._conn = self._open_checked()
            except (sqlite3.DatabaseError, _NeedsRecovery) as exc:
                if isinstance(exc, sqlite3.DatabaseError) and not _is_corruption(exc):
                    raise
                if self._conn is not None:
                    self._conn.close()
                self._conn = None
                if self._in_memory:
                    raise
                self.recovery = self._recover_locked(reason=f"{type(exc).__name__}: {exc}")
                self._conn = self._open_checked(after_recovery=True)

    def _open_checked(self, after_recovery: bool = False) -> sqlite3.Connection:
        existed = not self._in_memory and os.path.exists(self.path) and os.path.getsize(self.path) > 0
        conn = self._connect()
        self._conn = conn
        try:
            if self._check_on_open and existed and not after_recovery:
                res = [r[0] for r in conn.execute("PRAGMA quick_check").fetchall()]
                if res != ["ok"]:
                    raise _NeedsRecovery("quick_check failed: " + "; ".join(res[:5]))
            fresh = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()[0] == 0
            with self._tx() as c:
                for stmt in _split_sql(_SCHEMA):
                    c.execute(stmt)
                self._fts = self._setup_fts(c)
                if fresh:
                    self._set_meta(c, "schema_version", SCHEMA_VERSION)
                    self._set_meta(c, "created_at", _time.now())
                    self._set_meta(c, "db_uuid", uuid.uuid4().hex)
                if self._journal is not None and not after_recovery:
                    self._reconcile_journal(c, fresh)
                if self._get_meta(c, "schema_version") != SCHEMA_VERSION or (
                    self._fts and self._get_meta(c, "fts_stale") == "1"
                ):
                    self._rebuild_derived(c)
                    self._set_meta(c, "schema_version", SCHEMA_VERSION)
        except Exception:
            conn.close()
            self._conn = None
            raise
        return conn

    def _setup_fts(self, c: sqlite3.Connection) -> bool:
        if self._want_fts is False:
            exists = c.execute("SELECT 1 FROM sqlite_master WHERE name='beliefs_fts'").fetchone()
            if exists:
                self._set_meta(c, "fts_stale", "1")
            return False
        try:
            c.execute(_FTS_SCHEMA)
            return True
        except sqlite3.OperationalError:
            if self._want_fts:
                raise
            if c.execute("SELECT 1 FROM sqlite_master WHERE name='beliefs_fts'").fetchone():
                self._set_meta(c, "fts_stale", "1")
            return False

    # ----------------------------------------------------------- transactions
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, with busy retry. Every write goes through here."""
        with self._lock:
            conn = self._require_conn()
            deadline = time.monotonic() + self.busy_timeout
            delay = 0.005
            while True:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as e:
                    if not _is_busy(e) or time.monotonic() > deadline:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 0.25)
            self._tx_journaled = []
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if self._tx_journaled and self._journal is not None:
                    # Neutralise journal lines for events that never committed.
                    try:
                        self._journal.append([{"t": "void", "id": i} for i in self._tx_journaled])
                    except OSError:
                        pass
                raise
            finally:
                self._tx_journaled = []

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._require_conn()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Memory is closed")
        return self._conn

    @staticmethod
    def _get_meta(c: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def _set_meta(c: sqlite3.Connection, key: str, value: Any) -> None:
        c.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    # ------------------------------------------------------------ event log
    def _append_event(self, c: sqlite3.Connection, kind: str, subject: str, body: dict[str, Any]) -> dict[str, Any]:
        last = c.execute("SELECT seq, hash, recorded_at FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        now = _time.now()
        ev = {
            "seq": (last["seq"] + 1) if last else 1,
            "id": uuid.uuid4().hex,
            "kind": kind,
            "subject": subject,
            "body_json": canonical_json(body),
            # never let transaction time run backwards, even if the wall clock does
            "recorded_at": max(now, last["recorded_at"]) if last else now,
            "prev_hash": last["hash"] if last else GENESIS,
        }
        ev["hash"] = event_hash(ev)
        c.execute(
            "INSERT INTO events(seq, id, kind, subject, body, recorded_at, prev_hash, hash) VALUES(?,?,?,?,?,?,?,?)",
            (ev["seq"], ev["id"], ev["kind"], ev["subject"], ev["body_json"], ev["recorded_at"],
             ev["prev_hash"], ev["hash"]),
        )
        if self._journal is not None:
            self._tx_journaled.append(ev["id"])
            end = self._journal.append([event_to_record(ev)])
            self._set_meta(c, "journal_offset", end)
            self._set_meta(c, "journal_seq", ev["seq"])
        return ev

    def _events_for(self, c: sqlite3.Connection, subject: str) -> list[dict[str, Any]]:
        rows = c.execute("SELECT * FROM events WHERE subject=? ORDER BY seq", (subject,)).fetchall()
        return [_with_body(_row_to_event(r)) for r in rows]

    def _write_derived(self, c: sqlite3.Connection, subject: str, events: list[dict[str, Any]]) -> None:
        beliefs, rels = compute_timeline(events)
        if self._fts:
            old = [r[0] for r in c.execute("SELECT seq FROM beliefs WHERE subject=?", (subject,))]
            c.executemany("DELETE FROM beliefs_fts WHERE rowid=?", [(s,) for s in old])
        c.execute("DELETE FROM beliefs WHERE subject=?", (subject,))
        c.execute("DELETE FROM relations WHERE subject=?", (subject,))
        placeholders = ",".join("?" * len(_BELIEF_COLS))
        c.executemany(
            f"INSERT INTO beliefs({','.join(_BELIEF_COLS)}) VALUES({placeholders})",
            [tuple(canonical_json(b[k]) if k == "metadata" else b[k] for k in _BELIEF_COLS) for b in beliefs],
        )
        c.executemany(
            "INSERT OR IGNORE INTO relations(src, dst, kind, subject) VALUES(?,?,?,?)",
            [(s, d, k, subject) for s, d, k in rels],
        )
        if self._fts:
            c.executemany(
                "INSERT INTO beliefs_fts(rowid, subject, body) VALUES(?,?,?)",
                [(b["seq"], b["subject"], b["value"] if b["value"] == b["text"] else f"{b['value']} {b['text']}")
                 for b in beliefs],
            )

    def _refresh_subject(self, c: sqlite3.Connection, subject: str) -> None:
        self._write_derived(c, subject, self._events_for(c, subject))

    def _rebuild_derived(self, c: sqlite3.Connection) -> int:
        c.execute("DELETE FROM beliefs")
        c.execute("DELETE FROM relations")
        if self._fts:
            c.execute("DELETE FROM beliefs_fts")
            self._set_meta(c, "fts_stale", "0")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in c.execute("SELECT * FROM events ORDER BY seq"):
            ev = _with_body(_row_to_event(row))
            if ev["subject"] is not None:
                grouped.setdefault(ev["subject"], []).append(ev)
        for subject, evs in grouped.items():
            self._write_derived(c, subject, evs)
        return c.execute("SELECT count(*) FROM beliefs").fetchone()[0]

    # ---------------------------------------------------- journal reconcile
    def _reconcile_journal(self, c: sqlite3.Connection, fresh: bool) -> None:
        j = self._journal
        assert j is not None
        size = j.size()
        db_max = c.execute("SELECT coalesce(max(seq), 0) FROM events").fetchone()[0]
        if fresh and size > 0:
            scan = j.scan(0)
            if scan.resolved():
                raise _NeedsRecovery("database is missing or empty but the journal has events")
        offset = int(self._get_meta(c, "journal_offset", "0") or 0)
        if size < offset:
            # Journal was truncated/replaced: regenerate it from the (healthy) database.
            evs = [_row_to_event(r) for r in c.execute("SELECT * FROM events ORDER BY seq")]
            end = j.rewrite(evs, backup_suffix=f".bak-{_stamp()}")
            self._set_meta(c, "journal_offset", end)
            self._set_meta(c, "journal_seq", db_max)
            return
        scan = j.scan(offset)
        tail = scan.resolved()
        missing = [
            ev for ev in tail
            if not c.execute("SELECT 1 FROM events WHERE id=?", (ev["id"],)).fetchone()
        ]
        appended_void = False
        if missing:
            if len(missing) == 1 and missing[0]["seq"] == db_max + 1:
                # The one in-flight write of a crashed process: it never committed.
                j.append([{"t": "void", "id": missing[0]["id"]}])
                appended_void = True
            else:
                raise _NeedsRecovery(f"{len(missing)} journaled event(s) missing from the database")
        jseq = int(self._get_meta(c, "journal_seq", "0") or 0)
        seen = {ev["id"] for ev in scan.events}
        extra = [
            _row_to_event(r)
            for r in c.execute("SELECT * FROM events WHERE seq > ? ORDER BY seq", (jseq,))
            if r["id"] not in seen
        ]
        if extra:
            j.append([event_to_record(ev) for ev in extra])
        if extra or appended_void or scan.end_offset != offset or jseq != db_max:
            self._set_meta(c, "journal_offset", j.size())
            self._set_meta(c, "journal_seq", db_max)

    # ------------------------------------------------------------- recovery
    def recover(self, reason: str = "manual") -> dict[str, Any]:
        """Force a full recovery: move the DB aside and rebuild it from salvage + journal."""
        if self._in_memory:
            raise RecoveryError("cannot recover an in-memory database")
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            with _FileLock(self._lock_path):
                self.recovery = self._recover_locked(reason)
                self._conn = self._open_checked(after_recovery=True)
            return self.recovery

    def _recover_locked(self, reason: str) -> dict[str, Any]:
        stamp = _stamp()
        backup = None
        salvaged: list[dict[str, Any]] = []
        salvage_error = None
        if any(os.path.exists(self.path + s) for s in ("", "-wal")):
            backup = f"{self.path}.corrupt-{stamp}"
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(self.path + suffix):
                    os.replace(self.path + suffix, backup + suffix)
            jmax = 0
            if self._journal is not None:
                jmax = max((e["seq"] for e in self._journal.scan(0).events), default=0)
            salvaged, salvage_error = _salvage(backup, probe_hint=jmax)

        scan = self._journal.scan(0) if self._journal is not None else None
        from_journal = scan.resolved() if scan else []

        merged: dict[int, dict[str, Any]] = {ev["seq"]: ev for ev in from_journal}
        for ev in salvaged:  # committed DB rows win over journal lines for the same seq
            merged[ev["seq"]] = ev
        ids: set[str] = set()
        events: list[dict[str, Any]] = []
        for seq in sorted(merged):
            ev = merged[seq]
            if ev["id"] in ids:
                continue
            ids.add(ev["id"])
            events.append(ev)

        breaks = []
        prev = GENESIS
        for ev in events:
            if ev["prev_hash"] != prev:
                breaks.append(ev["seq"])
            prev = ev["hash"]

        conn = self._connect()
        self._conn = conn
        try:
            with self._tx() as c:
                for stmt in _split_sql(_SCHEMA):
                    c.execute(stmt)
                self._fts = self._setup_fts(c)
                c.executemany(
                    "INSERT INTO events(seq, id, kind, subject, body, recorded_at, prev_hash, hash) VALUES(?,?,?,?,?,?,?,?)",
                    [(e["seq"], e["id"], e["kind"], e["subject"], e["body_json"], e["recorded_at"],
                      e["prev_hash"], e["hash"]) for e in events],
                )
                self._set_meta(c, "schema_version", SCHEMA_VERSION)
                self._set_meta(c, "created_at", _time.now())
                self._set_meta(c, "db_uuid", uuid.uuid4().hex)
                self._set_meta(c, "acknowledged_breaks", json.dumps(breaks))
                self._rebuild_derived(c)
                end = 0
                if self._journal is not None:
                    end = self._journal.rewrite(events, backup_suffix=f".bak-{stamp}")
                self._set_meta(c, "journal_offset", end)
                self._set_meta(c, "journal_seq", events[-1]["seq"] if events else 0)
                report = {
                    "recovered_at": _time.now(),
                    "reason": reason,
                    "backup": backup,
                    "events_recovered": len(events),
                    "from_database": len(salvaged),
                    "from_journal": len(from_journal),
                    "journal_bad_lines": scan.bad_lines if scan else 0,
                    "salvage_error": salvage_error,
                    "chain_breaks": breaks,
                }
                self._set_meta(c, "last_recovery", json.dumps(report))
        finally:
            conn.close()
            self._conn = None
        return report

    # ================================================================ public
    def remember(
        self,
        text: str | None = None,
        *,
        subject: str | None = None,
        polarity: int | None = None,
        value: str | None = None,
        source: str = "user",
        confidence: float | None = None,
        valid_from: _time.TimeLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Belief]:
        """Record one or more stances. Returns the resulting beliefs (one per claim).

        * ``remember("don't do A")`` – the extractor finds ``subject='a', polarity=-1``.
        * ``remember(subject="A", polarity=+1, value="A is good")`` – fully explicit.
        * Explicit ``subject`` / ``polarity`` / ``confidence`` override extraction.
        """
        if text is None and subject is None and value is None:
            raise ValueError("remember() needs text, or subject/value")
        if polarity is not None and polarity not in (-1, 0, 1):
            raise ValueError("polarity must be -1, 0 or +1")
        if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        vf = _time.normalize(valid_from)

        claims: list[Claim]
        if subject is not None:
            norm = normalize_subject(subject)
            if not norm:
                raise ValueError("subject is empty after normalisation")
            pol = polarity
            conf = confidence
            if (pol is None or conf is None) and text:
                found = self.extractor.extract(text)
                match = next((cl for cl in found if cl.subject == norm), found[0] if len(found) == 1 else None)
                if match is not None:
                    pol = match.polarity if pol is None else pol
                    conf = match.confidence if conf is None else conf
            claims = [Claim(norm, pol or 0, value if value is not None else (text or subject),
                            1.0 if conf is None else float(conf))]
        else:
            src_text = text if text is not None else str(value)
            claims = [c for c in self.extractor.extract(src_text) if normalize_subject(c.subject)]
            if not claims:
                raise ValueError("could not extract a subject from text; pass subject= explicitly")
            claims = [
                Claim(normalize_subject(c.subject), c.polarity if polarity is None else polarity,
                      value if value is not None else c.value,
                      c.confidence if confidence is None else float(confidence))
                for c in claims
            ]

        raw_text = text if text is not None else (value or "")
        ids: list[str] = []
        with self._tx() as c:
            for cl in claims:
                body = {
                    "polarity": cl.polarity,
                    "value": cl.value,
                    "text": raw_text,
                    "source": source,
                    "confidence": cl.confidence,
                    "valid_from": vf,
                    "metadata": metadata or {},
                }
                ev = self._append_event(c, "assert", cl.subject, body)
                ids.append(ev["id"])
                self._refresh_subject(c, cl.subject)
            rows = [c.execute("SELECT * FROM beliefs WHERE id=?", (i,)).fetchone() for i in ids]
        return [Belief._from_row(r) for r in rows]

    def current(self, subject: str, at: _time.TimeLike | None = None) -> Belief | None:
        """The belief about ``subject`` that is valid now (or at valid-time ``at``)."""
        key = normalize_subject(subject)
        t = _time.normalize(at) or _time.now()
        with self._read() as c:
            row = c.execute(
                "SELECT * FROM beliefs WHERE subject=? AND forgotten=0 AND valid_from<=? "
                "AND (valid_to IS NULL OR valid_to>?) ORDER BY valid_from DESC, seq DESC LIMIT 1",
                (key, t, t),
            ).fetchone()
        return Belief._from_row(row) if row else None

    def as_of(self, subject: str, at: _time.TimeLike, known_at: _time.TimeLike | None = None) -> Belief | None:
        """Time travel. ``at`` is valid time ("what was true then");
        ``known_at`` is transaction time ("using only what had been recorded by then")."""
        if known_at is None:
            return self.current(subject, at=at)
        key = normalize_subject(subject)
        t, k = _time.normalize(at), _time.normalize(known_at)
        assert t is not None and k is not None
        with self._read() as c:
            events = [e for e in self._events_for(c, key) if e["recorded_at"] <= k]
        beliefs, _ = compute_timeline(events)
        best = pick_valid(beliefs, t)
        return Belief._from_row(best) if best else None

    def all_current(self, limit: int | None = None) -> list[Belief]:
        """Every subject's current belief, most recently recorded first."""
        t = _time.now()
        sql = (
            "SELECT * FROM beliefs WHERE forgotten=0 AND valid_from<=? AND (valid_to IS NULL OR valid_to>?) "
            "ORDER BY recorded_at DESC, seq DESC"
        )
        args: list[Any] = [t, t]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._read() as c:
            return [Belief._from_row(r) for r in c.execute(sql, args)]

    def history(self, subject: str, include_forgotten: bool = True) -> list[Belief]:
        """All beliefs ever recorded about ``subject``, in valid-time order."""
        key = normalize_subject(subject)
        sql = "SELECT * FROM beliefs WHERE subject=?"
        if not include_forgotten:
            sql += " AND forgotten=0"
        with self._read() as c:
            rows = c.execute(sql + " ORDER BY valid_from, seq", (key,)).fetchall()
        return [Belief._from_row(r) for r in rows]

    def relations(self, subject: str) -> list[dict[str, str]]:
        key = normalize_subject(subject)
        with self._read() as c:
            rows = c.execute(
                "SELECT r.src, r.dst, r.kind FROM relations r JOIN beliefs b ON b.id=r.src "
                "WHERE r.subject=? ORDER BY b.seq, r.kind", (key,)
            ).fetchall()
        return [{"src": r["src"], "dst": r["dst"], "kind": r["kind"]} for r in rows]

    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        include_superseded: bool = False,
        include_forgotten: bool = False,
    ) -> list[Belief]:
        """Relevant beliefs ranked by relevance (BM25) blended with recency.

        ``score = (1 - w) * relevance/max_relevance + w * 0.5 ** (age / half_life)``,
        scaled by ``0.75 + 0.25 * confidence``. By default only current
        (non-superseded, non-forgotten) beliefs are returned.
        """
        terms = query_terms(query)
        if not terms or k <= 0:
            return []
        now = _time.now()
        filters, args = [], []
        if not include_forgotten:
            filters.append("b.forgotten=0")
        if not include_superseded:
            filters.append("(b.valid_to IS NULL OR b.valid_to>?)")
            args.append(now)
        where = (" AND " + " AND ".join(filters)) if filters else ""
        with self._read() as c:
            if self._fts:
                match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
                rows = c.execute(
                    "SELECT b.*, -bm25(beliefs_fts, 2.0, 1.0) AS rel FROM beliefs_fts "
                    f"JOIN beliefs b ON b.seq=beliefs_fts.rowid WHERE beliefs_fts MATCH ?{where} "
                    "ORDER BY bm25(beliefs_fts, 2.0, 1.0) LIMIT ?",
                    [match, *args, max(200, k * 20)],
                ).fetchall()
                scored = [(r, float(r["rel"])) for r in rows if r["rel"] > 0]
            else:
                rows = c.execute(f"SELECT * FROM beliefs b WHERE 1=1{where}", args).fetchall()
                docs = [tokenize(r["subject"]) * 2 + tokenize(f"{r['value']} {r['text']}") for r in rows]
                scored = [(r, s) for r, s in zip(rows, bm25_scores(terms, docs)) if s > 0]
        if not scored:
            return []
        top = max(s for _, s in scored)
        half_life = max(self.half_life_days, 1e-9) * 86400.0
        w = min(max(self.recency_weight, 0.0), 1.0)
        out = []
        for r, rel in scored:
            recency = 0.5 ** (_time.age_seconds(r["recorded_at"], now) / half_life)
            score = ((1 - w) * rel / top + w * recency) * (0.75 + 0.25 * float(r["confidence"]))
            out.append(Belief._from_row(r, score=round(score, 6)))
        out.sort(key=lambda b: (b.score or 0.0, b.seq), reverse=True)
        return out[:k]

    def forget(
        self,
        target: str | None = None,
        *,
        id: str | None = None,
        subject: str | None = None,
        reason: str = "",
        source: str = "user",
    ) -> list[Belief]:
        """Soft-forget a belief (by id) or a whole subject. Logged as an event; nothing is deleted.

        ``target`` is tried as a belief id (exact, or unique prefix of >= 8 chars)
        and otherwise treated as a subject. Returns the beliefs that were
        forgotten (empty if there was nothing active to forget).
        """
        if sum(x is not None for x in (target, id, subject)) != 1:
            raise ValueError("pass exactly one of target, id= or subject=")
        with self._tx() as c:
            row = None
            if id is not None or target is not None:
                ident = (id if id is not None else target).strip()  # type: ignore[union-attr]
                row = c.execute("SELECT * FROM beliefs WHERE id=?", (ident,)).fetchone()
                if row is None and len(ident) >= 8 and all(ch in "0123456789abcdef" for ch in ident.lower()):
                    cand = c.execute("SELECT * FROM beliefs WHERE id LIKE ? LIMIT 2", (ident.lower() + "%",)).fetchall()
                    row = cand[0] if len(cand) == 1 else None
                if row is None and id is not None:
                    raise KeyError(f"no belief with id {id!r}")
            if row is not None:
                if row["forgotten"]:
                    return []
                key = row["subject"]
                affected = [row["id"]]
                body = {"target_id": row["id"], "reason": reason, "source": source}
            else:
                key = normalize_subject(subject if subject is not None else target)  # type: ignore[arg-type]
                affected = [r[0] for r in c.execute("SELECT id FROM beliefs WHERE subject=? AND forgotten=0", (key,))]
                if not affected:
                    return []
                body = {"target_id": None, "reason": reason, "source": source}
            self._append_event(c, "forget", key, body)
            self._refresh_subject(c, key)
            rows = [c.execute("SELECT * FROM beliefs WHERE id=?", (i,)).fetchone() for i in affected]
        return [Belief._from_row(r) for r in rows]

    def subjects(self, limit: int | None = None) -> list[str]:
        sql = "SELECT subject FROM beliefs WHERE forgotten=0 GROUP BY subject ORDER BY max(seq) DESC"
        with self._read() as c:
            rows = c.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        return [r[0] for r in rows]

    def events(self, subject: str | None = None) -> list[dict[str, Any]]:
        """Raw events (the append-only log), oldest first."""
        with self._read() as c:
            if subject is None:
                rows = c.execute("SELECT * FROM events ORDER BY seq").fetchall()
            else:
                rows = c.execute("SELECT * FROM events WHERE subject=? ORDER BY seq",
                                 (normalize_subject(subject),)).fetchall()
        out = []
        for r in rows:
            ev = _with_body(_row_to_event(r))
            ev.pop("body_json")
            out.append(ev)
        return out

    def rebuild(self) -> int:
        """Drop and recompute the derived belief view from the event log. Returns belief count."""
        with self._tx() as c:
            return self._rebuild_derived(c)

    def check(self) -> dict[str, Any]:
        """Verify integrity: SQLite quick_check, per-event hashes, hash chain,
        derived view == replay of events, and journal coverage."""
        with self._tx() as c:  # blocks writers -> consistent snapshot of DB + journal
            qc = [r[0] for r in c.execute("PRAGMA quick_check").fetchall()]
            evs = [_row_to_event(r) for r in c.execute("SELECT * FROM events ORDER BY seq")]
            bad_hash = [e["seq"] for e in evs if event_hash(e) != e["hash"]]
            acknowledged = set(json.loads(self._get_meta(c, "acknowledged_breaks", "[]") or "[]"))
            breaks, prev = [], GENESIS
            for e in evs:
                if e["prev_hash"] != prev and e["seq"] not in acknowledged:
                    breaks.append(e["seq"])
                prev = e["hash"]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for e in evs:
                grouped.setdefault(e["subject"], []).append(_with_body(e))
            mismatched = []
            for subject, sevs in grouped.items():
                expected, _ = compute_timeline(sevs)
                actual = [dict(r) for r in c.execute("SELECT * FROM beliefs WHERE subject=? ORDER BY seq", (subject,))]
                for a in actual:
                    a["metadata"] = json.loads(a["metadata"] or "{}")
                exp = sorted(expected, key=lambda b: b["seq"])
                if [tuple(b[k] for k in _BELIEF_COLS) for b in exp] != [tuple(a[k] for k in _BELIEF_COLS) for a in actual]:
                    mismatched.append(subject)
            orphan = c.execute(
                "SELECT count(*) FROM beliefs WHERE subject NOT IN (SELECT DISTINCT subject FROM events)"
            ).fetchone()[0]
            report: dict[str, Any] = {
                "path": self.path,
                "quick_check": qc,
                "events": len(evs),
                "beliefs": c.execute("SELECT count(*) FROM beliefs").fetchone()[0],
                "bad_hashes": bad_hash,
                "chain_breaks": breaks,
                "acknowledged_breaks": sorted(acknowledged),
                "derived_mismatch": mismatched + (["<orphan beliefs>"] if orphan else []),
                "fts": self._fts,
            }
            if self._journal is not None:
                scan = self._journal.scan(0)
                jids = {e["id"] for e in scan.events if e["id"] not in scan.voids}
                db_ids = {e["id"] for e in evs}
                report["journal"] = {
                    "path": self.journal_path,
                    "events": len(scan.resolved()),
                    "bad_lines": scan.bad_lines,
                    "voided": len(scan.voids),
                    "missing_from_journal": len(db_ids - jids),
                    "not_in_database": len({e["id"] for e in scan.resolved()} - db_ids),
                }
            report["last_recovery"] = json.loads(self._get_meta(c, "last_recovery", "null") or "null")
        jr = report.get("journal", {})
        report["ok"] = (
            qc == ["ok"] and not bad_hash and not breaks and not report["derived_mismatch"]
            and not jr.get("missing_from_journal") and not jr.get("not_in_database")
        )
        return report

    def stats(self) -> dict[str, Any]:
        with self._read() as c:
            return {
                "path": self.path,
                "events": c.execute("SELECT count(*) FROM events").fetchone()[0],
                "beliefs": c.execute("SELECT count(*) FROM beliefs").fetchone()[0],
                "current": len(self.all_current()),
                "subjects": c.execute("SELECT count(DISTINCT subject) FROM beliefs").fetchone()[0],
                "fts": self._fts,
                "journal": self.journal_path,
            }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Memory({self.path!r})"


# --------------------------------------------------------------------------- #
# salvage
# --------------------------------------------------------------------------- #

def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _split_sql(script: str) -> list[str]:
    """Split the schema into statements (triggers contain ';' inside BEGIN...END)."""
    out, buf = [], []
    for line in script.splitlines():
        if line.strip().startswith("--") or not line.strip():
            continue
        buf.append(line)
        joined = "\n".join(buf).strip()
        if joined.endswith(";") and (not joined.upper().startswith("CREATE TRIGGER") or joined.upper().endswith("END;")):
            out.append(joined)
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def _salvage(db_path: str, probe_hint: int = 0) -> tuple[list[dict[str, Any]], str | None]:
    """Best-effort extraction of hash-valid events from a damaged database file.

    Works on a temporary copy so the backup stays byte-for-byte untouched.
    Tries a full scan first; if that fails part-way, probes rowids one by one.
    """
    tmpdir = tempfile.mkdtemp(prefix="stancelog-salvage-")
    error: str | None = None
    found: dict[int, dict[str, Any]] = {}
    try:
        copy = os.path.join(tmpdir, "salvage.db")
        for suffix in ("", "-wal"):
            if os.path.exists(db_path + suffix):
                shutil.copyfile(db_path + suffix, copy + suffix)
        if not os.path.exists(copy):
            return [], "no database file"
        try:
            conn = sqlite3.connect(copy, isolation_level=None)
        except sqlite3.Error as e:
            return [], str(e)
        conn.row_factory = sqlite3.Row
        cols = "seq, id, kind, subject, body, recorded_at, prev_hash, hash"

        def take(row: sqlite3.Row) -> None:
            try:
                ev = _row_to_event(row)
                if event_hash(ev) == ev["hash"]:
                    found[ev["seq"]] = ev
            except (TypeError, KeyError, IndexError, AttributeError):
                pass

        try:
            for row in conn.execute(f"SELECT {cols} FROM events ORDER BY seq"):
                take(row)
        except sqlite3.Error as e:
            error = f"full scan failed: {e}"
            try:
                hi = conn.execute("SELECT max(seq) FROM events").fetchone()[0] or 0
            except sqlite3.Error:
                hi = 0
            limit = max(hi, probe_hint) + 1000
            misses = 0
            for seq in range(1, limit + 1):
                if seq in found:
                    continue
                try:
                    row = conn.execute(f"SELECT {cols} FROM events WHERE seq=?", (seq,)).fetchone()
                except sqlite3.Error:
                    row = None
                if row is None:
                    misses += 1
                    if misses > 1000 and seq > probe_hint:
                        break
                    continue
                misses = 0
                take(row)
        finally:
            conn.close()
    except Exception as e:  # salvage must never make recovery fail
        error = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return [found[s] for s in sorted(found)], error
