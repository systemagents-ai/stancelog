import json

import pytest

from stancelog import Claim, Extractor, HeuristicExtractor, LLMExtractor, Memory, normalize_subject

H = HeuristicExtractor()


def one(text):
    claims = H.extract(text)
    assert len(claims) == 1, claims
    return claims[0].subject, claims[0].polarity


@pytest.mark.parametrize("text,subject", [
    ("don't do A", "a"),
    ("Don't do A.", "a"),
    ("do not use X", "x"),
    ("never use X", "x"),
    ("Never use eval()", "eval"),
    ("avoid A", "a"),
    ("Avoid using global state", "global state"),
    ("stop using moment.js anymore", "moment.js"),
    ("We should not deploy on Fridays", "deploy on fridays"),
    ("A is bad", "a"),
    ("mocking the database is a bad idea", "mocking the database"),
    ("I hate tabs", "tabs"),
    ("A isn't good", "a"),
    ("no semicolons", "semicolons"),
])
def test_negative(text, subject):
    assert one(text) == (subject, -1)


@pytest.mark.parametrize("text,subject", [
    ("maybe A is better", "a"),
    ("Maybe A is better after all", "a"),
    ("use A", "a"),
    ("prefer A", "a"),
    ("A is good", "a"),
    ("A is really great", "a"),
    ("Always use type hints", "type hints"),
    ("I prefer dark mode", "dark mode"),
    ("we should use postgres", "postgres"),
    ("Actually, the A approach is the way to go", "a approach"),
    ("no, A is better", "a"),
    ("go with C++", "c++"),
])
def test_positive(text, subject):
    assert one(text) == (subject, 1)


def test_hedged_confidence():
    assert H.extract("maybe A is better")[0].confidence < 1.0
    assert H.extract("A is better")[0].confidence == 1.0


def test_comparatives_yield_two_claims():
    c = {(x.subject, x.polarity) for x in H.extract("prefer pnpm over npm")}
    assert c == {("pnpm", 1), ("npm", -1)}
    c = {(x.subject, x.polarity) for x in H.extract("use B instead of A")}
    assert c == {("b", 1), ("a", -1)}
    c = {(x.subject, x.polarity) for x in H.extract("postgres is better than mysql")}
    assert c == {("postgres", 1), ("mysql", -1)}


def test_multiple_clauses():
    c = {(x.subject, x.polarity) for x in H.extract("Don't use tabs, use spaces. Also avoid eval")}
    assert c == {("tabs", -1), ("spaces", 1), ("eval", -1)}


def test_neutral_facts_and_fallback():
    [c] = H.extract("editor: vim")
    assert (c.subject, c.polarity, c.value) == ("editor", 0, "vim")
    [c] = H.extract("My favourite editor is Helix")
    assert (c.subject, c.polarity, c.value) == ("favourite editor", 0, "Helix")
    [c] = H.extract("the sky")
    assert (c.subject, c.polarity) == ("sky", 0)
    assert HeuristicExtractor(fallback_note=False).extract("hello there") == []


def test_normalization():
    assert normalize_subject("A") == normalize_subject("a") == normalize_subject(" the A! ") == "a"
    assert normalize_subject("“Node.js”") == "node.js"
    assert normalize_subject("A anymore please") == "a"
    assert normalize_subject("C#") == "c#"
    assert normalize_subject("A") == "a"  # a bare article-like subject survives


def test_protocol_and_custom_extractor(db_path):
    class Upper:
        def extract(self, text):
            return [Claim(text.upper(), 1, text)]

    assert isinstance(H, Extractor) and isinstance(Upper(), Extractor)
    with Memory(db_path, extractor=Upper()) as m:
        [b] = m.remember("Thing")
        assert b.subject == "thing" and b.polarity == 1


def test_llm_extractor_json_string():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return 'Sure!\n```json\n[{"subject": "Kubernetes", "polarity": -1, "value": "too complex", "confidence": 0.7}]\n```'

    ex = LLMExtractor(fake_llm, known_subjects=lambda: ["docker"])
    [c] = ex.extract("k8s is overkill for us")
    assert c == Claim("kubernetes", -1, "too complex", 0.7)
    assert "k8s is overkill for us" in seen["prompt"] and '"docker"' in seen["prompt"]


def test_llm_extractor_structured_and_fallback():
    ex = LLMExtractor(lambda p: [{"subject": "A", "polarity": "positive"}])
    assert ex.extract("whatever") == [Claim("a", 1, "A", 1.0)]

    def broken(prompt):
        raise RuntimeError("model offline")

    ex = LLMExtractor(broken)
    assert ex.extract("don't do A") == [Claim("a", -1, "don't do A", 1.0)]
    assert isinstance(ex.last_error, RuntimeError)
    assert LLMExtractor(lambda p: "no json here", use_fallback=False).extract("x") == []


def test_llm_extractor_with_memory_gets_known_subjects(db_path):
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return json.dumps([{"subject": "a", "polarity": 1, "value": "A is better"}])

    with Memory(db_path, extractor=LLMExtractor(fake)) as m:
        m.remember("A first", subject="a", polarity=-1)
        [b] = m.remember("on reflection the first option wins")
        assert b.subject == "a" and b.contradicts is not None
        assert '"a"' in prompts[-1]
