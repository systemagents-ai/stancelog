from datetime import datetime, timedelta, timezone


def test_relevance_ranking(any_search_mem):
    m = any_search_mem
    m.remember("use pytest for testing python code")
    m.remember("prefer dark mode in the editor")
    m.remember("deploy target: prod-3")
    hits = m.recall("python testing")
    assert hits and "pytest" in hits[0].subject
    assert all(h.score is not None for h in hits)
    assert m.recall("kubernetes helm chart") == []
    assert m.recall("") == []


def test_recency_breaks_ties(any_search_mem, monkeypatch):
    m = any_search_mem
    from stancelog import _time

    old = datetime.now(timezone.utc) - timedelta(days=90)
    monkeypatch.setattr(_time, "now", lambda: _time.fmt(old))
    [a] = m.remember("database: postgres for the billing service", subject="billing db")
    monkeypatch.undo()
    [b] = m.remember("database: postgres for the search service", subject="search db")
    hits = m.recall("postgres database service", k=2)
    assert [h.id for h in hits] == [b.id, a.id]
    assert hits[0].score > hits[1].score


def test_superseded_and_forgotten_excluded(any_search_mem):
    m = any_search_mem
    [bad] = m.remember("don't use redis")
    [good] = m.remember("redis is fine for caching")
    assert good.contradicts == bad.id
    hits = m.recall("redis caching")
    assert [h.id for h in hits] == [good.id]
    assert {h.id for h in m.recall("redis caching", include_superseded=True)} == {bad.id, good.id}
    m.forget("redis")
    assert m.recall("redis") == []


def test_k_limit(any_search_mem):
    m = any_search_mem
    for i in range(10):
        m.remember(f"use tool{i} for logging", subject=f"logging tool {i}")
    assert len(m.recall("logging", k=3)) == 3


def test_stemming(any_search_mem):
    m = any_search_mem
    m.remember("never use tabs for indentation")
    [hit] = m.recall("tab")
    assert hit.subject == "tabs for indentation"
