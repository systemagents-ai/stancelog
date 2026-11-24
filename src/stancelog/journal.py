"""Append-only JSONL mirror of the event log (the recovery source of last resort).

Every event is written here (and ``fsync``-ed) *inside* the SQLite write
transaction, before ``COMMIT``. The journal is therefore a superset of the
committed events: at most one extra, never-committed "ghost" record can exist
per crash (the event that was being written when the process died). Ghosts are
neutralised on the next healthy open by appending a ``void`` record, and during
recovery by the rule "for duplicate ``seq`` the later line wins" (a later
writer can only reuse a seq if the earlier write never committed).

Line format (one JSON object per line)::

    {"t":"ev","seq":1,"id":"...","kind":"assert","subject":"a","body":"{...}",
     "recorded_at":"...","prev_hash":"...","hash":"..."}
    {"t":"void","id":"..."}

Every event line is self-verifying: ``hash`` is recomputed on read, so torn or
garbled lines are detected and skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .hashing import event_hash


@dataclass
class JournalScan:
    events: list[dict[str, Any]] = field(default_factory=list)  # valid event records, file order
    voids: set[str] = field(default_factory=set)
    bad_lines: int = 0
    end_offset: int = 0

    def resolved(self) -> list[dict[str, Any]]:
        """Events after removing voids and resolving duplicate seqs (last line wins)."""
        by_seq: dict[int, dict[str, Any]] = {}
        for ev in self.events:
            if ev["id"] in self.voids:
                continue
            by_seq[ev["seq"]] = ev
        return [by_seq[s] for s in sorted(by_seq)]


def event_to_record(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": "ev",
        "seq": ev["seq"],
        "id": ev["id"],
        "kind": ev["kind"],
        "subject": ev["subject"],
        "body": ev["body_json"],
        "recorded_at": ev["recorded_at"],
        "prev_hash": ev["prev_hash"],
        "hash": ev["hash"],
    }


def record_to_event(rec: dict[str, Any]) -> dict[str, Any] | None:
    try:
        ev = {
            "seq": int(rec["seq"]),
            "id": str(rec["id"]),
            "kind": str(rec["kind"]),
            "subject": rec["subject"],
            "body_json": str(rec["body"]),
            "recorded_at": str(rec["recorded_at"]),
            "prev_hash": str(rec["prev_hash"]),
            "hash": str(rec["hash"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if event_hash(ev) != ev["hash"]:
        return None
    return ev


class Journal:
    def __init__(self, path: str) -> None:
        self.path = path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except FileNotFoundError:
            return 0

    def append(self, records: list[dict[str, Any]]) -> int:
        """Append records durably; returns the new file size.

        Callers must hold the database write lock (BEGIN IMMEDIATE), which
        serialises appends across threads and processes.
        """
        new_file = not os.path.exists(self.path)
        data = "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records)
        payload = data.encode("utf-8")
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        fd = os.open(self.path, flags, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size > 0:
                # A previous writer may have died mid-line; never glue our record onto a torn one.
                os.lseek(fd, size - 1, os.SEEK_SET)
                if os.read(fd, 1) != b"\n":
                    payload = b"\n" + payload
            view = memoryview(payload)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
            end = os.fstat(fd).st_size
        finally:
            os.close(fd)
        if new_file:
            _fsync_dir(os.path.dirname(os.path.abspath(self.path)))
        return end

    def scan(self, offset: int = 0) -> JournalScan:
        out = JournalScan(end_offset=offset)
        try:
            f = open(self.path, "rb")
        except FileNotFoundError:
            return out
        with f:
            f.seek(offset)
            pos = offset
            for raw in f:
                pos += len(raw)
                if not raw.endswith(b"\n"):
                    # torn tail (writer died mid-line): don't advance past it
                    out.bad_lines += 1
                    pos -= len(raw)
                    break
                line = raw.strip()
                if not line:
                    out.end_offset = pos
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    out.bad_lines += 1
                    out.end_offset = pos
                    continue
                if isinstance(rec, dict) and rec.get("t") == "void" and isinstance(rec.get("id"), str):
                    out.voids.add(rec["id"])
                elif isinstance(rec, dict) and rec.get("t") == "ev" and (ev := record_to_event(rec)):
                    out.events.append(ev)
                else:
                    out.bad_lines += 1
                out.end_offset = pos
        return out

    def rewrite(self, events: list[dict[str, Any]], backup_suffix: str | None = None) -> int:
        """Atomically replace the journal with exactly ``events`` (keeps a backup if asked)."""
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            for ev in events:
                f.write((json.dumps(event_to_record(ev), separators=(",", ":"), ensure_ascii=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        if backup_suffix and os.path.exists(self.path):
            os.replace(self.path, self.path + backup_suffix)
        os.replace(tmp, self.path)
        _fsync_dir(os.path.dirname(os.path.abspath(self.path)))
        return self.size()


def _fsync_dir(path: str) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
