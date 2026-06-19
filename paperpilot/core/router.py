"""Query intent router for retrieval and answer-generation policy.

Priority:
  1. Exact-match cache (0 ms) — avoids repeated LLM calls for identical queries
  2. LLM classifier (≈200 ms) — accurate intent detection via small model
  3. Keyword regex fallback — used when no LLM client is available

Cache is in-process (LRU, maxsize=256).  A production deployment would use
Redis with a short TTL; for the single-paper web app this is sufficient.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from paperpilot.core.classifier import classify_query, is_yes_no_question

_CROSS_TABLE_PROMPT_SYSTEM = (
    "You are a query classifier for a research paper Q&A system.\n"
    "Determine whether the question requires jointly reading data from "
    "multiple distinct tables to answer (not just one table or a paragraph).\n"
    "Answer with exactly one word: yes or no."
)

_INTENT_PROMPT_SYSTEM = """You classify user questions for a research-paper RAG system.
The question may be English, Chinese, or mixed.
Return JSON only. No markdown.

Schema:
{
  "query_type": "numerical|comparison|why_how|methodology|factual",
  "cross_table": boolean,
  "is_yes_no": boolean,
  "evidence_need": "single_chunk|multi_chunk_synthesis|method_section|table_or_numeric_sentence|two_sided_evidence|two_tables|supporting_and_contrary_evidence",
  "source_preference": ["paragraph"|"table"|"figure_caption"],
  "answer_style": "concise_cited|numeric_with_units|structured_comparison|yes_no_then_evidence|explanatory_synthesis|method_steps",
  "confidence_checks": string[],
  "comparison_entities": string[],
  "needs_calculation": boolean
}

Guidelines:
- numerical: asks for numbers, metrics, counts, scores, deltas, maxima/minima, rankings by metric, or quantitative improvement.
- comparison: compares models, methods, datasets, baselines, variants, papers, or asks which performs better.
- why_how: asks why/how/explain/summarize/contribution/limitation and needs synthesis.
- methodology: asks about architecture, training, objective, loss, encoder/decoder, embeddings, parameters, implementation.
- factual: asks for a direct fact not covered above.
- cross_table is true only when jointly reading multiple distinct tables is needed.
- For comparison, set answer_style structured_comparison and include require_two_sided_evidence.
- For yes/no, set answer_style yes_no_then_evidence even if query_type remains factual.
- For numerical calculations, set needs_calculation true and include must_find_numeric_value.
"""

_CROSS_TABLE_FALLBACK_RE = re.compile(
    r"(?i)\b(compared?\s+to|versus|vs\.?|higher\s+than|lower\s+than|"
    r"better\s+than|worse\s+than|outperform\w*|more\s+than|less\s+than|"
    r"difference\s+between|between\s+\S+\s+and)\b"
)

_NUMERICAL_REASONING_RE = re.compile(
    r"(?i)\b(highest|lowest|best|worst|largest|smallest|max(?:imum)?|"
    r"min(?:imum)?|average|mean|difference|gap|improv|increase|decrease|"
    r"higher|lower|more|less|exceed|surpass)\b"
    r"|最高|最低|最好|最差|最大|最小|平均|差距|提升|增加|减少|更高|更低|超过"
)

_SYNTHESIS_RE = re.compile(
    r"(?i)\b(summarize|summary|overall|main idea|contribution|limitation|"
    r"why|how|explain|motivat|advantage|disadvantage)\b"
    r"|总结|概括|整体|主要|贡献|局限|为什么|为何|如何|解释|优势|劣势"
)

_FIGURE_RE = re.compile(r"(?i)\b(figure|fig\.?|plot|diagram|chart)\b|图|图表")
_TABLE_RE = re.compile(
    r"(?i)\b(table|score|bleu|rouge|f1|accuracy|precision|recall|result)\b"
    r"|表|分数|准确率|精确率|召回率|结果|表现"
)


@dataclass(frozen=True)
class QueryPlan:
    """Structured routing output shared by retrieval and generation."""

    query_type: str
    cross_table: bool = False
    is_yes_no: bool = False
    evidence_need: str = "single_chunk"
    source_preference: list[str] = field(default_factory=lambda: ["paragraph", "table"])
    answer_style: str = "concise_cited"
    confidence_checks: list[str] = field(default_factory=list)
    comparison_entities: list[str] = field(default_factory=list)
    needs_calculation: bool = False
    classifier: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


def _comparison_entities(question: str) -> list[str]:
    patterns = [
        r"(?i)\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$|\s+on\s+|\s+in\s+|\s+for\s+)",
        r"(?i)\bcompare\s+(.+?)\s+(?:with|to|against|and)\s+(.+?)(?:\?|$|\s+on\s+|\s+in\s+|\s+for\s+)",
        r"(?i)\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|\s+on\s+|\s+in\s+|\s+for\s+)",
        r"(.+?)和(.+?)(?:相比|对比|比较|的区别|的差异|哪个|谁)",
    ]
    for pattern in patterns:
        m = re.search(pattern, question)
        if m:
            entities = [re.sub(r"\s+", " ", g).strip(" ,.;:") for g in m.groups()]
            return [e for e in entities if 1 < len(e) <= 80][:2]
    return []


@lru_cache(maxsize=256)
def _cached_llm_route(question: str, model: str, base_url: str | None,
                      api_key_hint: str) -> bool:
    """Inner cached call — keyed on (question, model, base_url, api_key_hint).

    api_key_hint is the first 8 chars of the key so different keys produce
    different cache entries without storing the full secret.
    """
    import openai
    client = openai.OpenAI(api_key=_FULL_KEY, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CROSS_TABLE_PROMPT_SYSTEM},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_tokens=5,
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer.startswith("y")


# Module-level slot so the full key is available inside the cached function
_FULL_KEY: str = ""

_QUERY_TYPES = {"numerical", "comparison", "why_how", "methodology", "factual"}
_EVIDENCE_NEEDS = {
    "single_chunk",
    "multi_chunk_synthesis",
    "method_section",
    "table_or_numeric_sentence",
    "two_sided_evidence",
    "two_tables",
    "supporting_and_contrary_evidence",
}
_SOURCE_PREFS = {"paragraph", "table", "figure_caption"}
_ANSWER_STYLES = {
    "concise_cited",
    "numeric_with_units",
    "structured_comparison",
    "yes_no_then_evidence",
    "explanatory_synthesis",
    "method_steps",
}


def _json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM intent response did not contain a JSON object")
    return json.loads(text[start:end + 1])


@lru_cache(maxsize=512)
def _cached_llm_intent(question: str, model: str, base_url: str | None,
                       api_key_hint: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=_FULL_KEY, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _INTENT_PROMPT_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=350,
    )
    return _json_object(resp.choices[0].message.content or "")


def _as_bool(value, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_str_list(value, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item:
            continue
        if allowed is not None and item not in allowed:
            continue
        if item not in out:
            out.append(item)
    return out


def _coerce_llm_plan(raw: dict, fallback: QueryPlan) -> QueryPlan:
    query_type = raw.get("query_type")
    if query_type not in _QUERY_TYPES:
        query_type = fallback.query_type

    evidence_need = raw.get("evidence_need")
    if evidence_need not in _EVIDENCE_NEEDS:
        evidence_need = fallback.evidence_need

    answer_style = raw.get("answer_style")
    if answer_style not in _ANSWER_STYLES:
        answer_style = fallback.answer_style

    source_preference = _as_str_list(raw.get("source_preference"), _SOURCE_PREFS)
    if not source_preference:
        source_preference = fallback.source_preference

    checks = _as_str_list(raw.get("confidence_checks"))
    if not checks:
        checks = fallback.confidence_checks

    entities = _as_str_list(raw.get("comparison_entities"))
    if not entities:
        entities = fallback.comparison_entities
    entities = [e[:80] for e in entities][:4]

    # Keep critical consistency rules deterministic after the LLM vote.
    is_yes_no = _as_bool(raw.get("is_yes_no"), fallback.is_yes_no)
    cross_table = _as_bool(raw.get("cross_table"), fallback.cross_table)
    needs_calculation = _as_bool(raw.get("needs_calculation"), fallback.needs_calculation)

    if is_yes_no:
        answer_style = "yes_no_then_evidence"
        evidence_need = "supporting_and_contrary_evidence"
        if "state_yes_no_first" not in checks:
            checks.append("state_yes_no_first")

    if query_type == "numerical":
        answer_style = "numeric_with_units"
        evidence_need = "table_or_numeric_sentence"
        if "must_find_numeric_value" not in checks:
            checks.append("must_find_numeric_value")
        if "table" not in source_preference:
            source_preference = ["table"] + source_preference

    if query_type == "comparison":
        answer_style = "structured_comparison"
        if cross_table:
            evidence_need = "two_tables"
            if "require_two_tables" not in checks:
                checks.append("require_two_tables")
        else:
            evidence_need = "two_sided_evidence"
        if "require_two_sided_evidence" not in checks:
            checks.append("require_two_sided_evidence")

    return QueryPlan(
        query_type=query_type,
        cross_table=cross_table,
        is_yes_no=is_yes_no,
        evidence_need=evidence_need,
        source_preference=source_preference,
        answer_style=answer_style,
        confidence_checks=checks,
        comparison_entities=entities,
        needs_calculation=needs_calculation,
        classifier="llm",
    )


def route_cross_table(question: str, llm_client=None) -> bool:
    """Return True if the query needs cross-table joint retrieval.

    Args:
        question:   User's question string.
        llm_client: An openai.OpenAI-compatible client.  When None, falls back
                    to keyword regex (instant but less accurate).
    """
    if llm_client is None:
        return bool(_CROSS_TABLE_FALLBACK_RE.search(question))

    import os
    global _FULL_KEY
    _FULL_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    model     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url  = os.environ.get("OPENAI_BASE_URL")
    key_hint  = _FULL_KEY[:8]

    try:
        return _cached_llm_route(question, model, base_url, key_hint)
    except Exception:
        # Network / quota failure → fall back to regex, never crash
        return bool(_CROSS_TABLE_FALLBACK_RE.search(question))


def _rule_route_query(question: str, llm_client=None) -> QueryPlan:
    """Deterministic fallback plan for offline and LLM-failure cases."""
    query_type = classify_query(question)
    cross_table = route_cross_table(question, llm_client=llm_client)
    yes_no = is_yes_no_question(question)
    source_preference = ["paragraph", "table"]
    checks: list[str] = []
    entities = _comparison_entities(question)
    needs_calculation = False

    if _FIGURE_RE.search(question):
        source_preference = ["figure_caption", "paragraph", "table"]
        checks.append("warn_if_figure_missing")
    elif query_type == "numerical" or _TABLE_RE.search(question):
        source_preference = ["table", "paragraph"]
    elif query_type in {"why_how", "methodology"}:
        source_preference = ["paragraph", "table"]

    if query_type == "numerical":
        checks.extend(["must_find_numeric_value", "prefer_table_evidence"])
        needs_calculation = bool(_NUMERICAL_REASONING_RE.search(question))
        evidence_need = "table_or_numeric_sentence"
        answer_style = "numeric_with_units"
    elif query_type == "comparison":
        checks.append("require_two_sided_evidence")
        if cross_table:
            checks.append("require_two_tables")
            evidence_need = "two_tables"
        else:
            evidence_need = "two_sided_evidence"
        answer_style = "structured_comparison"
    elif yes_no:
        checks.extend(["state_yes_no_first", "cite_supporting_evidence"])
        evidence_need = "supporting_and_contrary_evidence"
        answer_style = "yes_no_then_evidence"
    elif query_type == "why_how" or _SYNTHESIS_RE.search(question):
        checks.append("synthesize_multiple_chunks")
        evidence_need = "multi_chunk_synthesis"
        answer_style = "explanatory_synthesis"
    elif query_type == "methodology":
        checks.append("prefer_method_sections")
        evidence_need = "method_section"
        answer_style = "method_steps"
    else:
        evidence_need = "single_chunk"
        answer_style = "concise_cited"

    return QueryPlan(
        query_type=query_type,
        cross_table=cross_table,
        is_yes_no=yes_no,
        evidence_need=evidence_need,
        source_preference=source_preference,
        answer_style=answer_style,
        confidence_checks=checks,
        comparison_entities=entities,
        needs_calculation=needs_calculation,
        classifier="rules",
    )


def route_query(question: str, llm_client=None) -> QueryPlan:
    """Return the full policy plan for this question.

    Uses an LLM intent classifier when credentials are configured, then falls
    back to deterministic rules on missing credentials, network errors, quota
    errors, or malformed model output.
    """
    if llm_client is None:
        return _rule_route_query(question, llm_client=None)

    import os
    global _FULL_KEY
    _FULL_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    key_hint = _FULL_KEY[:8]

    # Build the fallback without an LLM cross-table call so a successful intent
    # classification costs one model request, not two.
    fallback = _rule_route_query(question, llm_client=None)
    try:
        raw = _cached_llm_intent(question, model, base_url, key_hint)
        return _coerce_llm_plan(raw, fallback)
    except Exception:
        return _rule_route_query(question, llm_client=llm_client)
