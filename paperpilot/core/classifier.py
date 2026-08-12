"""Lightweight query type classifier.

Priority:
  numerical > comparison > methodology > why_how > factual

The router can still use an LLM planner, but this classifier is the deterministic
fallback used by local demos, tests, and benchmark/debug scripts.
"""
import re


_NUMERICAL_RE = re.compile(
    r"(?i)\b(how\s+many|how\s+much|bleu|rouge|f1|accuracy|precision|recall"
    r"|score|percent|percentage|number\s+of|size\s+of|ratio|perplexity"
    r"|training\s+time|dataset\s+size|batch\s+size|learning\s+rate|dropout\s+rate"
    r"|latency|memory\s+usage|throughput|error\s+rate|win\s+rate|improvement"
    r"|experimental\s+results?|key\s+results?|main\s+results?|performance\s+results?)\b"
    r"|多少|几个|分数|准确率|精确率|召回率|比例|百分比|提升"
)

_COMPARISON_RE = re.compile(
    r"(?i)\b(compar\w*|differ\w*|difference\s+between|versus|vs\.?|better\s+than"
    r"|worse\s+than|which\s+performs\s+better|performs\s+better|outperform"
    r"|is\s+.+?\s+and\s+.+?\s+better"
    r"|against|one\s+side|over\s+the\s+other|trade[- ]?off"
    r"|strengths?\s+and\s+weaknesses?|between\s+.+?\s+and\s+.+?)\b"
    r"|比较|对比|相比|不同|区别|差异|优于|劣于|超过|基线|表现|更好|更差|哪个|谁"
)

_METHODOLOGY_RE = re.compile(
    r"(?i)\b(architect|model\s+architecture|method\s+architecture|train(?:ing)?\s+objective"
    r"|encoder|decoder|embed(?:ding)?|loss\s+function|objective|pretraining|fine[- ]?tuning|fine.?tun"
    r"|attention\s+layer|layer|parameter|positional\s+encoding|optimization\s+setup"
    r"|preprocessing\s+pipeline|inference\s+algorithm|beam\s+search|masking\s+strategy"
    r"|feed[- ]?forward\s+network|residual\s+connection|normalization\s+layer"
    r"|tokenization\s+method|implementation\s+details|components?\s+are\s+used"
    r"|step\s+by\s+step|implemented|role\s+does.+play)\b"
    r"|架构|训练|编码器|解码器|嵌入|损失|目标函数|预训练|微调|参数|注意力"
)

_SYNTHESIS_RE = re.compile(
    r"(?i)\b(why|summari[sz]e|summary|explain|plain\s+terms|core\s+contribution"
    r"|main\s+contribution|contribution|main\s+idea|motivation|limitation"
    r"|main\s+finding|overall\s+approach|central\s+claim|research\s+problem"
    r"|significance|failure\s+mode|advantage|design\s+rationale|error\s+analysis"
    r"|discussion|conclusion|future\s+work|paper\s+is\s+about"
    r"|problem\s+the\s+paper\s+solves|evidence\s+supports)\b"
    r"|总结|概括|整体|主要|贡献|局限|为什么|为何|解释|优势|劣势"
)

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

QUERY_HINTS = {
    "why_how": "hint.why_how",
    "numerical": "hint.numerical",
}


def classify_query(question: str) -> str:
    if _NUMERICAL_RE.search(question):
        return "numerical"
    if _COMPARISON_RE.search(question):
        return "comparison"
    if _METHODOLOGY_RE.search(question):
        return "methodology"
    if _SYNTHESIS_RE.search(question):
        return "why_how"
    return "factual"


def is_yes_no_question(question: str) -> bool:
    return bool(_YES_NO_RE.match(question) or _YES_NO_ZH_RE.search(question))


def is_cross_table_query(question: str) -> bool:
    return bool(_CROSS_TABLE_RE.search(question))


def get_hint(query_type: str) -> str | None:
    return QUERY_HINTS.get(query_type)
