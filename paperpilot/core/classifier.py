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
        r"|多少|几个|分数|准确率|精确率|召回率|比例|百分比|提升"
    )),
    ("comparison", re.compile(
        r"(?i)\b(compar\w*|differ\w*|versus|vs\.?|better\s+than|worse\s+than"
        r"|outperform|baseline|against)\b"
        r"|比较|对比|相比|不同|区别|差异|优于|劣于|超过|基线|表现|更好|更差|哪个|谁"
    )),
    ("why_how", re.compile(r"(?i)(?:^\s*(?:why|how)\b)|为什么|为何|如何|怎么")),
    ("methodology", re.compile(
        r"(?i)\b(architect|train(?:ing)?|encoder|decoder|embed|loss"
        r"|objective|pretraining|fine.?tun|layer|attention|parameter)\b"
        r"|架构|训练|编码器|解码器|嵌入|损失|目标函数|预训练|微调|参数|注意力"
    )),
]

# PRD F-04: hints shown to user when retrieval is known to be limited
QUERY_HINTS = {
    "why_how":  "hint.why_how",
    "numerical": "hint.numerical",
}


_YES_NO_RE = re.compile(
    r"(?i)^\s*(is|are|does|do|did|has|have|was|were|can|could|would|will)\b"
    r"|^\s*(是否|是不是|能否|有没有|会不会)"
)
_YES_NO_ZH_RE = re.compile(r"是否|是不是|能否|有没有|会不会")

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
    return bool(_YES_NO_RE.match(question) or _YES_NO_ZH_RE.search(question))


def is_cross_table_query(question: str) -> bool:
    return bool(_CROSS_TABLE_RE.search(question))


def get_hint(query_type: str) -> str | None:
    return QUERY_HINTS.get(query_type)
