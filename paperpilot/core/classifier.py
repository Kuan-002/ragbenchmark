"""Query type classifier — identical priority chain as benchmark.

Priority order (must not change):
  numerical > comparison > why_how > methodology > factual

Benchmark note: why_how currently captures some methodology queries
(queries starting with 'how' that also contain methodology keywords).
This is a known issue; classifier will be updated in the next experiment round.
"""
import re

_RULES = [
    ("numerical", re.compile(
        r"(?i)\b(how\s+many|bleu|rouge|f1|accuracy|precision|recall"
        r"|score|percent|number\s+of|size\s+of|ratio)\b"
    )),
    ("comparison", re.compile(
        r"(?i)\b(compar|differ|versus|vs\.?|better\s+than|worse\s+than"
        r"|outperform|baseline|against)\b"
    )),
    ("why_how", re.compile(r"(?i)(?:^\s*(?:why|how)\b)")),
    ("methodology", re.compile(
        r"(?i)\b(architect|train(?:ing)?|encoder|decoder|embed|loss"
        r"|objective|pretraining|fine.?tun|layer|attention|parameter)\b"
    )),
]

# PRD F-04: hints shown to user when retrieval is known to be limited
QUERY_HINTS = {
    "why_how":  "hint.why_how",
    "numerical": "hint.numerical",
}


_YES_NO_RE = re.compile(
    r"(?i)^\s*(is|are|does|do|did|has|have|was|were|can|could|would|will)\b"
)

_CROSS_TABLE_RE = re.compile(
    r"(?i)\b(compared?\s+to|versus|vs\.?|higher\s+than|lower\s+than|"
    r"better\s+than|worse\s+than|outperform|more\s+than|less\s+than|"
    r"difference\s+between|between\s+\S+\s+and)\b"
)


def classify_query(question: str) -> str:
    for label, pattern in _RULES:
        if pattern.search(question):
            return label
    return "factual"


def is_yes_no_question(question: str) -> bool:
    return bool(_YES_NO_RE.match(question))


def is_cross_table_query(question: str) -> bool:
    return bool(_CROSS_TABLE_RE.search(question))


def get_hint(query_type: str) -> str | None:
    return QUERY_HINTS.get(query_type)
