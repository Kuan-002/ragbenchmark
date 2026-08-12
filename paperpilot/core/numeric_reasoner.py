"""Lightweight table and numeric reasoning for RAG answers.

This module does not try to be a full spreadsheet engine. It extracts
auditable numeric facts from retrieved chunks, then computes simple candidates
such as max/min/difference/average so the LLM has grounded numbers to verify.
"""
import re
from dataclasses import dataclass
from statistics import mean
from typing import Iterable


_NUM_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_HIGH_RE = re.compile(r"(?i)\b(highest|best|max(?:imum)?|largest|top)\b|最高|最好|最大")
_LOW_RE = re.compile(r"(?i)\b(lowest|worst|min(?:imum)?|smallest)\b|最低|最差|最小")
_DIFF_RE = re.compile(
    r"(?i)\b(difference|gap|delta|improv(?:e|ement)?|increase|decrease|higher|lower)\b"
    r"|差值|差距|提升|增加|减少|高多少|低多少"
)
_AVG_RE = re.compile(r"(?i)\b(average|mean)\b|平均")

_NUMERIC_STOPWORDS = {
    "what", "which", "how", "many", "much", "the", "a", "an", "is", "are",
    "was", "were", "of", "to", "in", "on", "for", "with", "by", "does",
    "do", "did", "they", "their", "paper", "table", "score", "result",
    "highest", "lowest", "best", "worst", "maximum", "minimum", "average",
    "mean", "difference", "gap", "improvement", "increase", "decrease",
}


@dataclass(frozen=True)
class NumericFact:
    source_rank: int
    value: float
    raw_value: str
    metric: str
    row_label: str
    caption: str
    chunk_id: str

    def label(self) -> str:
        parts = [p for p in [self.row_label, self.metric] if p]
        return " / ".join(parts) or self.caption or self.chunk_id

    def to_dict(self) -> dict:
        return {
            "source_rank": self.source_rank,
            "value": self.value,
            "raw_value": self.raw_value,
            "metric": self.metric,
            "row_label": self.row_label,
            "caption": self.caption,
            "chunk_id": self.chunk_id,
        }


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 2 and t not in _NUMERIC_STOPWORDS
    }


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _markdown_table_rows(text: str) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return [], []

    headers = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    return headers, body


def _structured_rows(chunk: dict) -> tuple[list[str], list[list[str]]]:
    columns = chunk.get("table_columns")
    rows = chunk.get("table_rows")
    if isinstance(columns, list) and isinstance(rows, list):
        clean_rows = [
            [str(cell).strip() for cell in row]
            for row in rows
            if isinstance(row, list) and any(str(cell).strip() for cell in row)
        ]
        if clean_rows:
            return [str(c).strip() for c in columns], clean_rows
    return _markdown_table_rows(chunk.get("text", ""))


def _facts_from_table(source_rank: int, chunk: dict) -> list[NumericFact]:
    caption = chunk.get("caption") or chunk.get("section", "")
    headers, rows = _structured_rows(chunk)
    facts: list[NumericFact] = []

    for row in rows:
        row_label = row[0] if row else ""
        for col_idx, cell in enumerate(row):
            for match in _NUM_RE.finditer(cell):
                value = _parse_number(match.group(0))
                if value is None:
                    continue
                metric = headers[col_idx] if col_idx < len(headers) else ""
                facts.append(NumericFact(
                    source_rank=source_rank,
                    value=value,
                    raw_value=match.group(0),
                    metric=metric,
                    row_label=row_label,
                    caption=caption,
                    chunk_id=chunk.get("chunk_id", ""),
                ))
    return facts


def _facts_from_text(source_rank: int, chunk: dict) -> list[NumericFact]:
    text = chunk.get("text", "")
    caption = chunk.get("caption") or chunk.get("section", "")
    facts: list[NumericFact] = []
    for match in _NUM_RE.finditer(text):
        value = _parse_number(match.group(0))
        if value is None:
            continue
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 90)
        context = re.sub(r"\s+", " ", text[start:end]).strip()
        facts.append(NumericFact(
            source_rank=source_rank,
            value=value,
            raw_value=match.group(0),
            metric=context[:140],
            row_label="",
            caption=caption,
            chunk_id=chunk.get("chunk_id", ""),
        ))
    return facts


def extract_numeric_facts(retrieved: list[dict]) -> list[NumericFact]:
    facts: list[NumericFact] = []
    for source_rank, result in enumerate(retrieved, 1):
        chunk = result.get("chunk", {})
        chunk_type = chunk.get("chunk_type")
        if chunk_type == "merged_tables":
            synthetic = dict(chunk)
            synthetic["chunk_type"] = "table"
            facts.extend(_facts_from_table(source_rank, synthetic))
        elif chunk_type == "table":
            facts.extend(_facts_from_table(source_rank, chunk))
        else:
            facts.extend(_facts_from_text(source_rank, chunk))
    return facts


def _relevance_score(fact: NumericFact, query_terms: set[str]) -> int:
    haystack = _tokens(" ".join([fact.metric, fact.row_label, fact.caption]))
    return len(query_terms.intersection(haystack))


def _relevant_facts(question: str, facts: Iterable[NumericFact]) -> list[NumericFact]:
    facts = list(facts)
    query_terms = _tokens(question)
    if not query_terms:
        return facts

    scored = [(fact, _relevance_score(fact, query_terms)) for fact in facts]
    max_score = max((score for _, score in scored), default=0)
    if max_score <= 0:
        return facts
    return [fact for fact, score in scored if score == max_score]


def numeric_reasoning_notes(question: str, retrieved: list[dict],
                            query_plan: dict | None = None) -> tuple[str, list[dict]]:
    wants_numeric = (query_plan or {}).get("query_type") == "numerical"
    wants_calc = bool((query_plan or {}).get("needs_calculation"))
    if not wants_numeric and not wants_calc:
        return "None.", []

    facts = extract_numeric_facts(retrieved)
    if not facts:
        return "No numeric values were extracted from the retrieved sources.", []

    selected = _relevant_facts(question, facts)
    notes = [
        f"Extracted {len(facts)} numeric values from retrieved sources; "
        f"{len(selected)} match the question terms best."
    ]

    if _HIGH_RE.search(question):
        best = max(selected, key=lambda f: f.value)
        notes.append(
            f"Candidate maximum: {best.raw_value} at {best.label()} "
            f"from source [{best.source_rank}]."
        )
    if _LOW_RE.search(question):
        best = min(selected, key=lambda f: f.value)
        notes.append(
            f"Candidate minimum: {best.raw_value} at {best.label()} "
            f"from source [{best.source_rank}]."
        )
    if _AVG_RE.search(question) and selected:
        notes.append(f"Candidate average over selected values: {mean(f.value for f in selected):g}.")
    if _DIFF_RE.search(question) and len(selected) >= 2:
        ordered = sorted(selected, key=lambda f: f.value, reverse=True)
        high, low = ordered[0], ordered[-1]
        notes.append(
            f"Candidate difference: {high.raw_value} - {low.raw_value} = "
            f"{high.value - low.value:g}, comparing {high.label()} "
            f"and {low.label()}."
        )

    if len(selected) <= 8:
        compact = "; ".join(
            f"[{f.source_rank}] {f.label()} = {f.raw_value}" for f in selected
        )
        notes.append(f"Relevant extracted values: {compact}.")

    notes.append("Treat these as candidates; the final answer must be supported by the cited source text.")
    return "\n".join(notes), [f.to_dict() for f in facts[:50]]
