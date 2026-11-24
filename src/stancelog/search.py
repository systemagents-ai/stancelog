"""Tokenisation and a pure-Python BM25 used when SQLite FTS5 is unavailable."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN = re.compile(r"[\w#+]+", re.UNICODE)
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been it this that with as at by "
    "do does did not no i we you me my our your should would could".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.casefold())]


def query_terms(text: str) -> list[str]:
    """Distinct query tokens, dropping stopwords unless that would leave nothing."""
    toks = list(dict.fromkeys(tokenize(text)))
    content = [t for t in toks if t not in _STOP]
    return content or toks


def stem(tok: str) -> str:
    for suf in ("ingly", "edly", "ing", "ies", "ed", "es", "ly", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def bm25_scores(query: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    q = [stem(t) for t in query]
    stemmed = [[stem(t) for t in d] for d in docs]
    n = len(stemmed)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in stemmed) / n or 1.0
    df: Counter[str] = Counter()
    for d in stemmed:
        df.update(set(d))
    scores = []
    for d in stemmed:
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores
