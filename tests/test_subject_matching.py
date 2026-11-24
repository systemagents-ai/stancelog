"""Natural-phrasing contradictions must land on one subject; unrelated subjects must not merge."""

import json
import subprocess
import sys

import pytest

from stancelog import Memory, subject_similarity

from test_interfaces import _env


@pytest.mark.parametrize("first,second", [
    ("don't use Redis for the cache", "maybe Redis is better for the cache"),
    ("never deploy on Fridays", "actually deploying on Fridays is fine"),
    ("avoid tabs", "prefer tabs"),
    ("maybe Redis is better for the cache", "don't use Redis for the cache"),  # reverse order
])
def test_natural_phrasings_share_one_timeline(mem, first, second):
    [old] = mem.remember(first)
    [new] = mem.remember(second)
    assert old.polarity * new.polarity == -1
    assert new.subject == old.subject  # attached to the existing subject
    assert new.supersedes == old.id and new.contradicts == old.id

    current = mem.all_current()
    assert [b.id for b in current] == [new.id]  # exactly one current belief

    hist = mem.history(old.subject)
    assert [b.id for b in hist] == [old.id, new.id]
    assert [b.status for b in hist] == ["superseded", "current"]
    assert hist[0].superseded_by == new.id
    kinds = {(r["src"], r["dst"], r["kind"]) for r in mem.relations(old.subject)}
    assert kinds == {(new.id, old.id, "supersedes"), (new.id, old.id, "contradicts")}


def test_lookup_by_short_name_resolves_to_matched_subject(mem):
    [old] = mem.remember("don't use Redis for the cache")
    [new] = mem.remember("maybe Redis is better for the cache")
    assert old.subject == "redis for the cache"
    assert new.metadata["matched_from"] == "redis"
    assert mem.resolve_subject("redis") == "redis for the cache"
    assert mem.current("redis").id == new.id
    assert [b.id for b in mem.history("Redis")] == [old.id, new.id]
    assert len(mem.relations("redis")) == 2
    assert mem.check()["ok"]
    mem.rebuild()  # subject choice lives in the event log, so replay is identical
    assert mem.current("redis").id == new.id


@pytest.mark.parametrize("a,b", [
    ("use redis", "don't use postgres"),
    ("avoid redis", "use postgres"),
    ("use tabs", "use spaces"),
    ("avoid tabs", "prefer spaces"),
    ("never use tabs for indentation", "use spaces for indentation"),
    ("don't use Redis for the cache", "use postgres for the cache"),
    ("use python 2", "avoid python 3"),
])
def test_no_false_merge(mem, a, b):
    [x] = mem.remember(a)
    [y] = mem.remember(b)
    assert x.subject != y.subject
    assert y.supersedes is None and y.contradicts is None
    assert {c.id for c in mem.all_current()} == {x.id, y.id}


def test_similarity_rules():
    assert subject_similarity("redis", "redis for the cache") >= 0.5
    assert subject_similarity("deploy on fridays", "deploying on fridays") == 1.0
    assert subject_similarity("redis", "postgres") == 0.0
    assert subject_similarity("tabs", "spaces") == 0.0
    assert subject_similarity("redis", "redis cluster config") == 0.0  # too weak a subset
    assert subject_similarity("python", "python 3") == 0.0  # version tokens distinguish
    assert subject_similarity("the", "a") == 0.0  # no content tokens


def test_explicit_subject_bypasses_matching(mem):
    [a] = mem.remember("don't use Redis for the cache")
    [b] = mem.remember("redis is fine", subject="redis")
    assert b.subject == "redis" and b.supersedes is None
    assert "matched_from" not in b.metadata
    assert mem.current("redis").id == b.id  # exact subject wins on lookup
    assert mem.current("redis for the cache").id == a.id


def test_matching_can_be_disabled(db_path):
    with Memory(db_path, subject_matching=False) as m:
        [a] = m.remember("don't use Redis for the cache")
        [b] = m.remember("maybe Redis is better for the cache")
        assert a.subject != b.subject


def test_claims_in_one_message_do_not_merge_with_each_other(mem):
    mem.remember("use Redis for the cache")
    bs = mem.remember("use redis, avoid redis for the cache")
    assert sorted(b.subject for b in bs) == ["redis", "redis for the cache"]


def test_forget_by_short_name(mem):
    mem.remember("don't use Redis for the cache")
    mem.remember("maybe Redis is better for the cache")
    assert len(mem.forget("redis")) == 2
    assert mem.current("redis") is None


@pytest.mark.parametrize("use_fts", [True, False], ids=["fts5", "python-bm25"])
def test_recall_ranks_current_above_superseded(db_path, use_fts):
    with Memory(db_path, use_fts=use_fts) as m:
        [old] = m.remember("don't use Redis for the cache")  # confident
        [new] = m.remember("maybe Redis is better for the cache")  # hedged -> lower confidence
        assert [h.id for h in m.recall("redis")] == [new.id]
        hits = m.recall("redis", include_superseded=True)
        assert [h.id for h in hits] == [new.id, old.id]
        assert [h.status for h in hits] == ["current", "superseded"]
        assert hits[0].score > hits[1].score
        assert hits[1].to_dict()["status"] == "superseded"


def _cli(db, *args):
    p = subprocess.run([sys.executable, "-m", "stancelog", *args], capture_output=True, text=True,
                       env=_env(db), timeout=60)
    assert p.returncode == 0, (p.stdout, p.stderr)
    return p.stdout


def test_cli_redis_scenario(db_path):
    _cli(db_path, "remember", "don't use Redis for the cache")
    _cli(db_path, "remember", "maybe Redis is better for the cache")

    cur = json.loads(_cli(db_path, "current", "--json"))
    assert len(cur) == 1
    assert cur[0]["polarity"] == 1 and cur[0]["status"] == "current"
    assert "redis" in cur[0]["subject"]
    human = _cli(db_path, "current")
    assert human.count("CURRENT") == 1

    assert json.loads(_cli(db_path, "current", "redis", "--json"))["id"] == cur[0]["id"]

    hist = json.loads(_cli(db_path, "history", "redis", "--json"))
    old, new = hist["history"]
    assert (old["polarity"], new["polarity"]) == (-1, 1)
    assert old["status"] == "superseded" and old["superseded_by"] == new["id"]
    assert new["contradicts"] == old["id"]
    assert {"src": new["id"], "dst": old["id"], "kind": "contradicts"} in hist["relations"]

    out = _cli(db_path, "history", "redis")
    assert f"SUPERSEDED by {new['id'][:8]}" in out
    assert f"contradicts {old['id'][:8]}" in out
    assert f"{new['id'][:8]} [+] CONTRADICTS {old['id'][:8]} [-]" in out
    assert out.index("[-] redis") < out.index("[+] redis")  # oldest first

    hits = json.loads(_cli(db_path, "recall", "redis", "--all", "--json"))
    assert [h["id"] for h in hits] == [new["id"], old["id"]]
