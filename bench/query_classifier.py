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
    ("why_how", re.compile(
        r"(?i)(?:^\s*(?:why|how)\b)|为什么|为何|如何|怎么"
    )),
    ("methodology", re.compile(
        r"(?i)\b(architect|train(?:ing)?|encoder|decoder|embed|loss"
        r"|objective|pretraining|fine.?tun|layer|attention|parameter)\b"
        r"|架构|训练|编码器|解码器|嵌入|损失|目标函数|预训练|微调|参数|注意力"
    )),
]


def classify_query(question: str) -> str:
    for label, pattern in _RULES:
        if pattern.search(question):
            return label
    return "factual"


def classify_all(queries):
    return {q["qid"]: classify_query(q["question"]) for q in queries}
