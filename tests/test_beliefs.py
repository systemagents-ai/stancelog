"""Belief semantics: supersession, contradictions, history, bi-temporal queries, forget."""

from datetime import datetime, timedelta, timezone

import pytest

from stancelog import Memory


def test_a_bad_then_a_good(mem):
    """The canonical scenario from the spec."""
    [bad] = mem.remember("Don't do A")
    [good] = mem.remember("Maybe A is better")

    assert bad.subject == good.subject == "a"
    assert bad.polarity == -1 and good.polarity == 1

    cur = mem.current("A")  # case-insensitive subject lookup
    assert cur is not None
    assert cur.id == good.id
    assert cur.stance == "positive"
    assert cur.confidence < 1.0  # "maybe" is hedged

    hist = mem.history("a")
    assert [b.id for b in hist] == [bad.id, good.id]
    assert [b.stance for b in hist] == ["negative", "positive"]
    old, new = hist
    assert old.valid_to == new.valid_from  # old belief closed when the new one began
    assert old.superseded_by == new.id
    assert new.supersedes == old.id
    assert new.contradicts == old.id
    assert new.valid_to is None

    rels = {(r["src"], r["dst"], r["kind"]) for r in mem.relations("a")}
    assert rels == {(new.id, old.id, "supersedes"), (new.id, old.id, "contradicts")}

    # nothing destroyed: both events still in the append-only log
    assert [e["body"]["polarity"] for e in mem.events("a")] == [-1, 1]


def test_same_polarity_supersedes_without_contradiction(mem):
    [a] = mem.remember("use pytest")
    [b] = mem.remember("prefer pytest")
    assert b.supersedes == a.id
    assert b.contradicts is None
    assert [r["kind"] for r in mem.relations("pytest")] == ["supersedes"]


def test_neutral_value_update(mem):
    mem.remember("deploy target: prod-1")
    mem.remember("deploy target: prod-3")
    cur = mem.current("deploy target")
    assert cur.value == "prod-3" and cur.polarity == 0
    assert len(mem.history("deploy target")) == 2


def test_explicit_subject_and_polarity(mem):
    [b] = mem.remember("tabs are evil", subject="Tabs", polarity=-1, source="test", confidence=0.8)
    assert (b.subject, b.polarity, b.source, b.confidence) == ("tabs", -1, "test", 0.8)
    # polarity inferred from text when not given
    [c] = mem.remember("use tabs", subject="tabs")
    assert c.polarity == 1 and c.contradicts == b.id


def test_validation(mem):
    with pytest.raises(ValueError):
        mem.remember()
    with pytest.raises(ValueError):
        mem.remember("x", polarity=2)
    with pytest.raises(ValueError):
        mem.remember("x", confidence=1.5)


def test_events_are_append_only(mem):
    import sqlite3

    mem.remember("never use eval")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        mem._conn.execute("UPDATE events SET subject='x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        mem._conn.execute("DELETE FROM events")


def test_bitemporal_valid_time_and_as_of(mem):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem.remember("avoid mongodb", valid_from=t0)
    mem.remember("mongodb is fine", valid_from=t0 + timedelta(days=10))
    assert mem.as_of("mongodb", t0 + timedelta(days=5)).polarity == -1
    assert mem.as_of("mongodb", t0 + timedelta(days=15)).polarity == 1
    assert mem.as_of("mongodb", t0 - timedelta(days=1)) is None
    assert mem.current("mongodb").polarity == 1


def test_backdated_fact_does_not_override_newer(mem):
    now = datetime.now(timezone.utc)
    [new] = mem.remember("use uv", valid_from=now - timedelta(days=1))
    [old] = mem.remember("don't use uv", valid_from=now - timedelta(days=30))  # learned late, true earlier
    assert mem.current("uv").id == new.id
    hist = mem.history("uv")
    assert [b.id for b in hist] == [old.id, new.id]
    assert hist[0].valid_to == new.valid_from


def test_transaction_time_as_of(mem):
    [first] = mem.remember("don't do A")
    known = first.recorded_at
    mem.remember("A is good")
    # what did we believe (valid now) using only knowledge recorded up to `known`?
    assert mem.as_of("a", datetime.now(timezone.utc), known_at=known).polarity == -1
    assert mem.as_of("a", datetime.now(timezone.utc), known_at=datetime.now(timezone.utc)).polarity == 1


def test_soft_forget_subject(mem):
    mem.remember("don't do A")
    mem.remember("A is good")
    gone = mem.forget("A", reason="user asked")
    assert len(gone) == 2 and all(b.forgotten for b in gone)
    assert mem.current("a") is None
    hist = mem.history("a")
    assert len(hist) == 2 and all(b.forgotten and b.forgotten_reason == "user asked" for b in hist)
    assert mem.history("a", include_forgotten=False) == []
    kinds = [e["kind"] for e in mem.events("a")]
    assert kinds == ["assert", "assert", "forget"]  # logged, nothing deleted
    assert mem.recall("A") == []
    # forgetting again is a no-op and is not logged
    assert mem.forget("a") == []
    assert len(mem.events("a")) == 3
    # new statements after a forget start fresh
    mem.remember("avoid A")
    assert mem.current("a").polarity == -1


def test_forget_by_id_reverts_to_previous(mem):
    [bad] = mem.remember("don't do A")
    [good] = mem.remember("A is good")
    gone = mem.forget(good.id, reason="mistake")
    assert [b.id for b in gone] == [good.id]
    cur = mem.current("a")
    assert cur.id == bad.id and cur.valid_to is None
    # prefix ids work too
    [x] = mem.remember("use ruff")
    assert [b.id for b in mem.forget(x.id[:10])] == [x.id]
    with pytest.raises(KeyError):
        mem.forget(id="deadbeefdeadbeef")


def test_all_current_and_subjects(mem):
    mem.remember("use ruff")
    mem.remember("don't use black")
    mem.remember("black is fine")
    cur = {b.subject: b.polarity for b in mem.all_current()}
    assert cur == {"ruff": 1, "black": 1}
    assert set(mem.subjects()) == {"ruff", "black"}


def test_rebuild_matches_incremental(mem):
    mem.remember("don't do A")
    mem.remember("A is good")
    mem.remember("prefer pnpm over npm")
    [x] = mem.remember("editor: vim")
    mem.forget(x.id)
    before = [b.to_dict() for s in ("a", "pnpm", "npm", "editor") for b in mem.history(s)]
    mem._conn.execute("DELETE FROM beliefs")  # simulate a lost/garbled derived view
    mem.rebuild()
    after = [b.to_dict() for s in ("a", "pnpm", "npm", "editor") for b in mem.history(s)]
    assert before == after
    assert mem.check()["ok"]


def test_check_detects_derived_drift(mem):
    mem.remember("don't do A")
    mem._conn.execute("UPDATE beliefs SET polarity=1")
    rep = mem.check()
    assert not rep["ok"] and rep["derived_mismatch"] == ["a"]
    mem.rebuild()
    assert mem.check()["ok"]


def test_in_memory_store():
    with Memory(":memory:") as m:
        m.remember("don't do A")
        m.remember("A is better")
        assert m.current("a").polarity == 1
        assert m.check()["ok"]
