"""Many processes and threads writing at once: no lost writes, no broken chain."""

import os
import subprocess
import sys
import threading

from stancelog import Memory

N_PROCS = 4
N_THREADS = 4
PER_WRITER = 25

PROC_WRITER = r"""
import sys
from stancelog import Memory
db, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with Memory(db) as m:
    for i in range(n):
        m.remember(f"{tag} item {i}: v{i}", subject=f"{tag} item {i}")
print("done")
"""


def test_processes_and_threads_concurrently(db_path):
    Memory(db_path).close()  # create schema up front
    env = dict(os.environ)
    env.pop("STANCELOG_DB", None)
    procs = [
        subprocess.Popen([sys.executable, "-c", PROC_WRITER, db_path, f"p{i}", str(PER_WRITER)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for i in range(N_PROCS)
    ]

    shared = Memory(db_path)
    errors = []

    def shared_writer(tag):
        try:
            for i in range(PER_WRITER):
                shared.remember(f"{tag} item {i}: v{i}", subject=f"{tag} item {i}")
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    def own_writer(tag):
        try:
            with Memory(db_path) as m:
                for i in range(PER_WRITER):
                    m.remember(f"{tag} item {i}: v{i}", subject=f"{tag} item {i}")
                    m.recall(tag)  # interleave reads
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=shared_writer, args=(f"s{i}",)) for i in range(N_THREADS)]
    threads += [threading.Thread(target=own_writer, args=(f"o{i}",)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        assert out.strip() == "done"
    assert not errors, errors
    shared.close()

    expected = {f"{tag} item {i}" for tag in [f"p{i}" for i in range(N_PROCS)]
                + [f"s{i}" for i in range(N_THREADS)] + [f"o{i}" for i in range(N_THREADS)]
                for i in range(PER_WRITER)}
    with Memory(db_path) as m:
        evs = m.events()
        assert len(evs) == len(expected)
        assert {e["subject"] for e in evs} == expected
        assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
        rep = m.check()
        assert rep["ok"], rep
        assert rep["journal"]["events"] == len(expected)
        assert len(m.all_current()) == len(expected)


def test_concurrent_updates_to_same_subject_keep_a_linear_history(db_path):
    with Memory(db_path) as m:
        def flip(pol):
            for _ in range(20):
                m.remember("A", subject="a", polarity=pol)

        ts = [threading.Thread(target=flip, args=(p,)) for p in (1, -1, 1, -1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        hist = m.history("a")
        assert len(hist) == 80
        assert sum(b.valid_to is None for b in hist) == 1
        for prev, cur in zip(hist, hist[1:]):
            assert cur.supersedes == prev.id
            assert (cur.contradicts == prev.id) == (prev.polarity != cur.polarity)
        assert m.check()["ok"]
