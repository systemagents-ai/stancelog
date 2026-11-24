"""Turning free text into (subject, polarity, value) claims.

An *extractor* is anything with ``extract(text) -> list[Claim]``. Two are built
in:

* :class:`HeuristicExtractor` – fast, deterministic, regex based. Understands
  directives such as "don't do A", "never use X", "avoid A" (polarity -1) and
  "use A", "prefer A", "maybe A is better", "A is good" (polarity +1), plus
  neutral facts such as "editor: vim" or "my editor is vim" (polarity 0).
* :class:`LLMExtractor` – wraps *your* callable (any LLM client, local model,
  or a stub). stancelog never makes network calls or needs an API key itself.

Subjects are normalised with :func:`normalize_subject` so that "A", "a" and
"the A" all land on the same key, which is what makes supersession and
contradiction detection work.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

__all__ = [
    "Claim",
    "Extractor",
    "HeuristicExtractor",
    "LLMExtractor",
    "normalize_subject",
]


@dataclass(frozen=True)
class Claim:
    """One extracted stance: ``polarity`` is +1 (endorse), -1 (reject) or 0 (neutral fact)."""

    subject: str
    polarity: int
    value: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.polarity not in (-1, 0, 1):
            raise ValueError(f"polarity must be -1, 0 or 1, got {self.polarity!r}")


@runtime_checkable
class Extractor(Protocol):
    """Protocol for pluggable extractors."""

    def extract(self, text: str) -> list[Claim]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_QUOTES = str.maketrans({c: "" for c in "\"'`‘’“”«»"})
_TRAILING_FILLER = re.compile(
    r"(?:\s+(?:anymore|any more|again|please|ever|at all|instead|for now|then|"
    r"after all|though|too|either|really|here|thanks|thank you))+$"
)
_LEADING_ARTICLE = re.compile(r"^(?:the|an|a)\s+(?=\S)")


def normalize_subject(subject: str) -> str:
    """Canonical key for a subject: case-folded, punctuation/article/filler stripped.

    Keeps characters that matter in technical names (``c++``, ``c#``,
    ``node.js``, ``gpt-4``, ``foo/bar``, ``snake_case``).
    """
    s = unicodedata.normalize("NFKC", subject).casefold().translate(_QUOTES)
    s = re.sub(r"[^\w\s\-./+#@:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .:-")
    s = _TRAILING_FILLER.sub("", s).strip(" .:-")  # fillers can stack ("A anymore please")
    return _LEADING_ARTICLE.sub("", s, count=1)  # once: "the A approach" -> "a approach"


# --------------------------------------------------------------------------- #
# Heuristic extractor
# --------------------------------------------------------------------------- #

_HEDGE_WORDS = (
    r"maybe|perhaps|probably|possibly|actually|i think|i guess|i suppose|i believe|"
    r"honestly|ok|okay|so|well|hmm+|now|on second thought|turns out|it turns out|"
    r"it seems|apparently|in fact|also|and|but|hey|note|fyi|btw|remember|"
    r"remember that|reminder|yes|yeah|nah|no(?=\s*,)|wait"
)
_HEDGE_PREFIX = re.compile(rf"^(?:(?:{_HEDGE_WORDS})\b[\s,:;-]*)+", re.IGNORECASE)
_UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|probably|possibly|i think|i guess|i suppose|might|may be|"
    r"could be|seems?|apparently|not sure)\b"
)

_ACTION = (
    r"(?:do|doing|use|using|try|trying|run|running|choose|choosing|pick|picking|"
    r"go with|going with|rely on|relying on|recommend|recommending|suggest|suggesting|"
    r"touch|add|adding|write|writing|call|calling|install|installing|enable|enabling)"
)
_SUBJECT_PRONOUN = r"(?:(?:you|we|i|they|one)\s+)?"

_NEG_BAD_WORDS = (
    r"bad|worse|worst|wrong|terrible|awful|horrible|broken|harmful|dangerous|"
    r"a bad idea|bad idea|a mistake|buggy|flaky|slow|unreliable|overrated|useless|"
    r"deprecated|insecure|problematic|not good|not great|not ok|not okay|"
    r"not recommended|not the way to go|not worth it|a no-go|banned|forbidden"
)
_POS_GOOD_WORDS = (
    r"good|better|best|great|fine|ok|okay|preferred|recommended|the way to go|"
    r"a good idea|good idea|correct|right|nice|excellent|ideal|the best option|"
    r"the right choice|worth it|superior|faster|safer|cleaner|allowed|acceptable|"
    r"the answer|the winner"
)
_COPULA = (
    r"(?:is|are|was|were|seems|seem|looks|look|feels|feel|sounds|sound|might be|"
    r"may be|could be|would be|should be|will be|has been|have been|becomes|is now|"
    r"are now|works|work)"
)
_INTENSIFIER = r"(?:(?:really|very|much|pretty|actually|kind of|kinda|quite|so|way|far|definitely)\s+)*"

_P = re.IGNORECASE

_BETTER = r"(?:better|faster|safer|cleaner|superior|nicer|simpler|preferable)"
_WORSE = r"(?:worse|slower|riskier|uglier|inferior|messier)"

# Ordered: first match wins. (pattern, polarity, kind)
_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    # --- comparisons: "A is better than B" -> +A, -B ---------------------------
    (re.compile(rf"^(?P<s>.+?)\s+{_COPULA}\s+{_INTENSIFIER}{_BETTER}\s+(?:than|to)\s+(?P<o>.+)$", _P), 1, "compare"),
    (re.compile(rf"^(?P<s>.+?)\s+{_COPULA}\s+{_INTENSIFIER}{_WORSE}\s+(?:than|to)\s+(?P<o>.+)$", _P), -1, "compare"),
    # --- negative directives ------------------------------------------------
    (re.compile(
        rf"^(?:please\s+)?{_SUBJECT_PRONOUN}(?:let'?s\s+not|don'?t|do not|dont|never|"
        rf"avoid|stop|quit|no longer|no more|shouldn'?t|should not|should never|"
        rf"must not|mustn'?t|must never|can'?t|cannot|can not|won'?t|will not|"
        rf"refuse to|ought not to)\s+(?:ever\s+)?(?:{_ACTION}\s+)?(?P<s>.+)$", _P), -1, "directive"),
    (re.compile(
        r"^(?:i|we)\s+(?:hate|dislike|don'?t like|do not like|don'?t want|do not want|"
        r"can'?t stand|regret using|regret)\s+(?P<s>.+)$", _P), -1, "preference"),
    (re.compile(
        rf"^(?P<s>.+?)\s+{_COPULA}\s+(?:a\s+)?{_INTENSIFIER}(?:{_NEG_BAD_WORDS})\b.*$", _P), -1, "judgement"),
    (re.compile(
        rf"^(?P<s>.+?)\s+(?:is not|isn'?t|are not|aren'?t|was not|wasn'?t|does not|"
        rf"doesn'?t|do not|don'?t)\s+(?:a\s+)?{_INTENSIFIER}(?:{_POS_GOOD_WORDS}|work|help|scale)\b.*$", _P),
     -1, "judgement"),
    (re.compile(r"^no\s+(?P<s>(?!longer\b|more\b).+)$", _P), -1, "directive"),
    # --- positive preferences ------------------------------------------------
    (re.compile(
        r"^(?:please\s+)?(?:(?:you|we|i)\s+(?:should|must|could|might|may)\s+(?:always\s+)?|"
        r"let'?s\s+|always\s+|do\s+)?(?:prefer|favou?r)\s+(?P<s>.+)$", _P), 1, "prefer"),
    (re.compile(r"^(?:i|we)\s+(?:like|love|prefer|want|recommend|enjoy|favou?r)\s+(?P<s>.+)$", _P), 1, "prefer"),
    (re.compile(
        rf"^(?P<s>.+?)\s+{_COPULA}\s+(?:a\s+)?{_INTENSIFIER}(?:{_POS_GOOD_WORDS})\b.*$", _P), 1, "judgement"),
    (re.compile(
        r"^(?:please\s+)?(?:(?:you|we|i)\s+(?:should|must|could|might|may)\s+(?:always\s+)?|"
        r"let'?s\s+|always\s+|do\s+)?(?:use|try|choose|pick|go with|switch to|"
        r"stick with|stick to|rely on|keep using|start using|recommend|adopt|enable|"
        r"go for|opt for)\s+(?P<s>.+)$", _P), 1, "directive"),
    (re.compile(r"^(?:you|we|i)\s+(?:should|must)\s+(?:always\s+)?(?P<s>.+)$", _P), 1, "directive"),
    (re.compile(r"^always\s+(?P<s>.+)$", _P), 1, "directive"),
    (re.compile(r"^(?:please\s+)?do\s+(?P<s>.+)$", _P), 1, "directive"),
    # --- neutral facts --------------------------------------------------------
    (re.compile(r"^(?P<s>[^:=]{1,80}?)\s*(?::|=|->)\s*(?P<v>\S.*)$", _P), 0, "kv"),
    (re.compile(r"^(?:my|our|the|their|his|her|your)\s+(?P<s>.+?)\s+(?:is|are|was|were)\s+(?P<v>.+)$", _P), 0, "fact"),
]

_PREFER_SPLIT = re.compile(r"\s+(?:over|instead of|rather than|not|vs\.?|versus)\s+", _P)
_INSTEAD_SPLIT = re.compile(r"\s+(?:instead of|rather than)\s+", _P)

# Split a message into clauses: sentence punctuation, newlines, and commas /
# "but" / "and" that introduce a new directive ("don't use A, use B instead").
_DIRECTIVE_START = (
    r"(?:use|prefer|don'?t|do not|never|avoid|always|stop|try|switch|stick|go with|"
    r"maybe|perhaps|actually|we should|you should|i prefer|let'?s)\b"
)
_CLAUSE_SPLIT = re.compile(
    rf"(?:[.;!?\n]+(?:\s+|$)|\s*,\s*(?:but\s+|and\s+|so\s+)?(?={_DIRECTIVE_START})|"
    rf"\s+(?:but|and|so|then)\s+(?={_DIRECTIVE_START}))",
    _P,
)


class HeuristicExtractor:
    """Deterministic regex-based extractor (no dependencies, no network).

    Each clause of the input yields at most one claim, except for comparative
    clauses ("use A instead of B", "prefer A over B") which yield +A and -B.

    Args:
        fallback_note: if True, text that matches no pattern becomes a neutral
            (polarity 0) note whose subject is the normalised text itself.
        hedged_confidence: confidence assigned to hedged claims ("maybe ...").
    """

    def __init__(self, *, fallback_note: bool = True, hedged_confidence: float = 0.6,
                 max_subject_len: int = 120) -> None:
        self.fallback_note = fallback_note
        self.hedged_confidence = hedged_confidence
        self.max_subject_len = max_subject_len

    # public -----------------------------------------------------------------
    def extract(self, text: str) -> list[Claim]:
        claims: list[Claim] = []
        for clause in self._clauses(text):
            claims.extend(self._extract_clause(clause))
        if not claims and self.fallback_note:
            subject = normalize_subject(text)[: self.max_subject_len].strip()
            if subject:
                claims.append(Claim(subject, 0, text.strip(), 1.0))
        # de-duplicate identical subjects within one message: last wins
        seen: dict[str, Claim] = {}
        for c in claims:
            seen.pop(c.subject, None)
            seen[c.subject] = c
        return list(seen.values())

    # internals --------------------------------------------------------------
    @staticmethod
    def _clauses(text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).replace("’", "'")
        parts = _CLAUSE_SPLIT.split(text)
        return [p.strip(" \t,") for p in parts if p and p.strip(" \t,")]

    def _extract_clause(self, clause: str) -> list[Claim]:
        confidence = self.hedged_confidence if _UNCERTAIN.search(clause.lower()) else 1.0
        body = _HEDGE_PREFIX.sub("", clause.strip()).strip()
        if not body:
            return []
        for pattern, polarity, kind in _PATTERNS:
            m = pattern.match(body)
            if not m:
                continue
            raw_subject = m.group("s")
            if polarity == 0:
                subject = normalize_subject(raw_subject)
                value = m.group("v").strip()
                if not subject or len(subject) > self.max_subject_len:
                    continue
                return [Claim(subject, 0, value, confidence)]
            out: list[Claim] = []
            if kind == "compare":
                a, b = normalize_subject(raw_subject), normalize_subject(m.group("o"))
                if a and len(a) <= self.max_subject_len:
                    out.append(Claim(a, polarity, clause.strip(), confidence))
                if b and b != a and len(b) <= self.max_subject_len:
                    out.append(Claim(b, -polarity, clause.strip(), confidence))
                if out:
                    return out
                continue
            splitter = _PREFER_SPLIT if kind == "prefer" else _INSTEAD_SPLIT
            pieces = splitter.split(raw_subject, maxsplit=1)
            main = normalize_subject(pieces[0])
            if not main or len(main) > self.max_subject_len:
                continue
            out.append(Claim(main, polarity, clause.strip(), confidence))
            if len(pieces) == 2:
                other = normalize_subject(pieces[1])
                if other and other != main and len(other) <= self.max_subject_len:
                    out.append(Claim(other, -polarity, clause.strip(), confidence))
            return out
        return []


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #

DEFAULT_LLM_PROMPT = """You extract durable memories for an AI agent.
Read the message and return ONLY a JSON array (no prose). Each element:
  {"subject": "<short lowercase noun phrase naming the thing being judged>",
   "polarity": 1 | -1 | 0,   // 1 = endorsed/recommended, -1 = rejected/discouraged, 0 = neutral fact
   "value": "<short restatement of the stance or fact>",
   "confidence": <number 0..1>}
Reuse an existing subject from KNOWN SUBJECTS when the message refers to the same thing.
Return [] if there is nothing worth remembering.

KNOWN SUBJECTS: {known_subjects}
MESSAGE: {text}
"""


class LLMExtractor:
    """Extractor backed by a user-supplied callable; stancelog itself makes no network calls.

    ``complete`` receives the rendered prompt (a ``str``) and may return a JSON
    string, a list of dicts / :class:`Claim`, or a single dict. Anything that
    can't be parsed falls back to ``fallback`` (a :class:`HeuristicExtractor`
    by default; pass ``None`` to get ``[]`` instead).

    ``known_subjects`` is an optional zero-arg callable returning existing
    subjects; :class:`~stancelog.Memory` wires it up automatically so the model
    can map paraphrases onto existing keys (which is what makes contradiction
    detection work across wording changes).
    """

    def __init__(
        self,
        complete: Callable[[str], Any],
        *,
        prompt: str = DEFAULT_LLM_PROMPT,
        fallback: Extractor | None = None,
        known_subjects: Callable[[], Iterable[str]] | None = None,
        use_fallback: bool = True,
    ) -> None:
        self.complete = complete
        self.prompt = prompt
        self.fallback = fallback if fallback is not None else (HeuristicExtractor() if use_fallback else None)
        self.known_subjects = known_subjects
        self.last_error: Exception | None = None

    def render(self, text: str) -> str:
        known = list(self.known_subjects() if self.known_subjects else [])[:200]
        return self.prompt.replace("{known_subjects}", json.dumps(known)).replace("{text}", text)

    def extract(self, text: str) -> list[Claim]:
        self.last_error = None
        try:
            raw = self.complete(self.render(text))
            claims = self._parse(raw)
        except Exception as exc:  # user callables can fail in any way
            self.last_error = exc
            claims = None
        if claims is None:
            return self.fallback.extract(text) if self.fallback else []
        return claims

    @staticmethod
    def _parse(raw: Any) -> list[Claim] | None:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            s = raw.strip()
            fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
            if fence:
                s = fence.group(1).strip()
            starts = [i for i in (s.find("["), s.find("{")) if i >= 0]
            if not starts:
                return None
            start = min(starts)
            end = max(s.rfind("]"), s.rfind("}"))
            raw = json.loads(s[start : end + 1])
        if isinstance(raw, dict):
            raw = raw.get("claims", raw.get("memories", [raw]))
        if isinstance(raw, Claim):
            raw = [raw]
        if not isinstance(raw, list):
            return None
        out: list[Claim] = []
        for item in raw:
            if isinstance(item, Claim):
                c = item
            elif isinstance(item, dict) and item.get("subject"):
                pol = item.get("polarity", 0)
                if isinstance(pol, str):
                    pol = {"positive": 1, "+": 1, "+1": 1, "1": 1, "negative": -1, "-": -1,
                           "-1": -1, "neutral": 0, "0": 0}.get(pol.strip().lower(), 0)
                pol = max(-1, min(1, int(round(float(pol)))))
                conf = float(item.get("confidence", 1.0))
                c = Claim(str(item["subject"]), pol, str(item.get("value") or item["subject"]),
                          max(0.0, min(1.0, conf)))
            else:
                continue
            subject = normalize_subject(c.subject)
            if subject:
                out.append(Claim(subject, c.polarity, c.value, c.confidence))
        return out
