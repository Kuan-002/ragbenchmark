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
    ("why_how", re.compile(
        r"(?i)(?:^\s*(?:why|how)\b)"
    )),
    ("methodology", re.compile(
        r"(?i)\b(architect|train(?:ing)?|encoder|decoder|embed|loss"
        r"|objective|pretraining|fine.?tun|layer|attention|parameter)\b"
    )),
]


def classify_query(question: str) -> str:
    for label, pattern in _RULES:
        if pattern.search(question):
            return label
    return "factual"


def classify_all(queries):
    return {q["qid"]: classify_query(q["question"]) for q in queries}
