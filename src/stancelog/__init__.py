"""stancelog – crash-proof, contradiction-aware memory for AI agents.

>>> from stancelog import Memory
>>> m = Memory(":memory:")
>>> _ = m.remember("Don't do A")
>>> _ = m.remember("Maybe A is better")
>>> m.current("a").stance
'positive'
>>> [b.stance for b in m.history("A")]
['negative', 'positive']
"""

from .extract import Claim, Extractor, HeuristicExtractor, LLMExtractor, normalize_subject
from .store import Belief, Memory, RecoveryError, default_db_path

__version__ = "0.1.0"

__all__ = [
    "Belief",
    "Claim",
    "Extractor",
    "HeuristicExtractor",
    "LLMExtractor",
    "Memory",
    "RecoveryError",
    "default_db_path",
    "normalize_subject",
    "__version__",
]
