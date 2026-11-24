"""Crash (kill -9), corruption, persistence and journal-reconciliation tests."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from stancelog import Memory

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")

WRITER = r"""
import sys
from stancelog import Memory
m = Memory(sys.argv[1])
start = int(sys.argv[2])
i = start
while True:
    m.remember(f"item {i} is good", subject=f"item {i}", source="writer")
    # Only acknowledge after remember() returned, i.e. after COMMIT.
    sys.stdout.write(f"{i}\n")
    sys.stdout.flush()
    i += 1
"""


def _spawn_writer(db, start):
    env = dict(os.environ)
    env.pop("STANCELOG_DB", None)
    return subprocess.Popen([sys.executable, "-c", WRITER, db, str(start)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)


def _kill_after(proc, n_acks, extra_delay):
    acked = []
    deadline = time.time() + 60
    while len(acked) < n_acks and time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        acked.append(int(line))
    time.sleep(extra_delay)  # land the kill at an arbitrary point mid-write
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(10)
    acked.extend(int(x) for x in proc.stdout.read().split())  # acks flushed before death
    assert acked, proc.stderr.read()
    return acked


def test_persistence_across_restarts(db_path):
    with Memory(db_path) as m:
        m.remember("don't do A")
        [good] = m.remember("A is better")
    with Memory(db_path) as m:
        assert m.current("a").id == good.id
        assert len(m.history("a")) == 2
        m.remember("never use eval")
    with Memory(db_path) as m:
        assert m.current("eval").polarity == -1
        assert m.check()["ok"]
        assert m.recovery is None


def test_kill9_mid_write_loses_nothing_committed(db_path):
    all_acked = []
    start = 0
    for round_, delay in enumerate((0.0, 0.003, 0.011)):
        proc = _spawn_writer(db_path, start)
        acked = _kill_after(proc, 15 + 5 * round_, delay)
        all_acked += acked
        start = max(acked) + 1000  # distinct subjects per round

        with Memory(db_path) as m:
            assert m.recovery is None, m.recovery  # a crash is not corruption; no recovery needed
            rep = m.check()
            assert rep["ok"], rep
            subjects = {e["subject"] for e in m.events()}
            # every acknowledged (committed) write is present...
            for i in all_acked:
                assert f"item {i}" in subjects
                assert m.current(f"item {i}").polarity == 1
            # ...and the log is contiguous, hash-valid, with no partial events
            seqs = [e["seq"] for e in m.events()]
            assert seqs == list(range(1, len(seqs) + 1))
            assert rep["bad_hashes"] == [] and rep["chain_breaks"] == []
            # at most one un-acked write (committed just before the kill) beyond the acks
            assert len(seqs) - len(all_acked) in (0, 1)
            all_acked = [int(s.split()[1]) for s in subjects]


def test_ghost_journal_entry_is_voided_not_resurrected(db_path):
    with Memory(db_path) as m:
        m.remember("use A")
        last = m.events()[-1]
    # Simulate a crash after the journal fsync but before COMMIT: a journal line with no DB row.
    from stancelog.hashing import canonical_json, event_hash
    from stancelog.journal import event_to_record

    ghost = {"seq": last["seq"] + 1, "id": "f" * 32, "kind": "assert", "subject": "ghost",
             "body_json": canonical_json({"polarity": 1, "value": "boo"}),
             "recorded_at": last["recorded_at"], "prev_hash": last["hash"]}
    ghost["hash"] = event_hash(ghost)
    with open(db_path + ".journal.jsonl", "a") as f:
        f.write(json.dumps(event_to_record(ghost)) + "\n")
        f.write('{"t":"ev","seq":99,"torn')  # plus a torn half-line from a dying writer

    with Memory(db_path) as m:
        assert m.recovery is None
        assert m.current("ghost") is None
        m.remember("use B")  # appends after the torn line safely
        rep = m.check()
        assert rep["ok"], rep
        assert rep["journal"]["voided"] == 1 and rep["journal"]["bad_lines"] >= 1

    _garble(db_path)
    with Memory(db_path) as m:  # recovery from the journal must not resurrect the ghost
        assert m.recovery is not None
        assert m.current("ghost") is None
        assert m.current("a").polarity == 1 and m.current("b").polarity == 1
        assert m.check()["ok"]


def _garble(path):
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.write(os.urandom(min(size, 4096)))  # destroys the header and first page(s)


def _fill(db_path, n=30):
    with Memory(db_path) as m:
        m.remember("don't do A")
        m.remember("A is better")
        for i in range(n):
            m.remember(f"fact {i}: value {i}")
        m.forget("fact 0")
        return [e["id"] for e in m.events()]


@pytest.mark.parametrize("damage", ["garble-header", "truncate-half", "truncate-zero", "delete", "random-pages"])
def test_corruption_recovery_from_journal(db_path, damage):
    ids = _fill(db_path)
    if damage == "garble-header":
        _garble(db_path)
    elif damage == "truncate-half":
        with open(db_path, "r+b") as f:
            f.truncate(os.path.getsize(db_path) // 2)
    elif damage == "truncate-zero":
        open(db_path, "wb").close()
    elif damage == "delete":
        os.remove(db_path)
    elif damage == "random-pages":
        size = os.path.getsize(db_path)
        with open(db_path, "r+b") as f:
            for off in range(4096, size, 8192):
                f.seek(off + 100)
                f.write(b"\xde\xad\xbe\xef" * 64)

    with Memory(db_path) as m:
        assert m.recovery is not None, "damage should have been detected"
        assert [e["id"] for e in m.events()] == ids  # every event back, in order
        assert m.current("a").polarity == 1
        assert [b.stance for b in m.history("a")] == ["negative", "positive"]
        assert m.current("fact 0") is None and m.history("fact 0")[0].forgotten
        assert m.check()["ok"], m.check()
        if damage != "delete":
            assert os.path.exists(m.recovery["backup"])  # damaged file kept for forensics
        m.remember("still writable")
    with Memory(db_path) as m:
        assert m.recovery is None
        assert m.current("still writable") is not None


def test_recovery_salvages_db_when_journal_is_lost(db_path):
    ids = _fill(db_path, n=5)
    os.remove(db_path + ".journal.jsonl")
    with Memory(db_path) as m:  # healthy DB regenerates its journal
        assert m.recovery is None
        assert m.check()["ok"]
    with Memory(db_path) as m:
        rep = m.recover(reason="test")
        assert rep["from_database"] == len(ids)
        assert [e["id"] for e in m.events()] == ids
        assert m.check()["ok"]


def test_recovery_merges_partial_db_and_journal(db_path):
    """Old DB copy + newer journal (e.g. restored backup): nothing is lost."""
    import shutil

    with Memory(db_path) as m:
        m.remember("don't do A")
    shutil.copy(db_path, db_path + ".old")
    with Memory(db_path) as m:
        m.remember("A is better")
        m.remember("use B")
    os.replace(db_path + ".old", db_path)
    for s in ("-wal", "-shm"):
        if os.path.exists(db_path + s):
            os.remove(db_path + s)
    with Memory(db_path) as m:
        assert m.recovery is not None
        assert m.current("a").polarity == 1 and m.current("b").polarity == 1
        assert len(m.events()) == 3 and m.check()["ok"]


def test_lost_journal_line_is_reported_as_acknowledged_break(db_path):
    _fill(db_path, n=3)
    jp = db_path + ".journal.jsonl"
    lines = open(jp).read().splitlines(keepends=True)
    del lines[2]  # one journal record lost
    open(jp, "w").writelines(lines)
    _garble(db_path)
    with Memory(db_path) as m:
        assert m.recovery["chain_breaks"] == [4]
        assert len(m.events()) == 5
        rep = m.check()
        assert rep["ok"] and rep["acknowledged_breaks"] == [4]
