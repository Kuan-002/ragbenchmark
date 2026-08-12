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
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

from paperpilot.core.classifier import classify_query, is_yes_no_question
from paperpilot.core.router_model import predict_query_type

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

_RETRIEVAL_PLANNER_PROMPT_SYSTEM = """You are the retrieval planner for a RAG system over scientific papers.
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
  "needs_calculation": boolean,
  "retrieval_strategy": "standard|fusion|hyde|decompose",
  "rewrite_queries": string[],
  "hyde_query": string
}

Planner rules:
- Use standard for direct single-fact questions.
- Use fusion for why/how, comparison, methodology, and ambiguous questions.
- Use hyde when the question is abstract and likely suffers vocabulary mismatch.
- Use decompose when the question asks for multiple entities, multiple criteria, or a comparison.
- rewrite_queries should be 2-4 short search queries, not answers.
- hyde_query should be a concise hypothetical relevant passage, not a final answer.
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
    r"(?i)\b(table|score|bleu|rouge|f1|accuracy|precision|recall|result|results|"
    r"experimental\s+results?|key\s+results?|main\s+results?|performance)\b"
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
    route_level: str = "rules"
    route_confidence: float = 0.0
    complexity_score: float = 0.0
    retrieval_strategy: str = "standard"
    rewrite_queries: list[str] = field(default_factory=list)
    hyde_query: str = ""

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
_RETRIEVAL_STRATEGIES = {"standard", "fusion", "hyde", "decompose"}


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


@lru_cache(maxsize=256)
def _cached_llm_retrieval_plan(question: str, model: str, base_url: str | None,
                               api_key_hint: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=_FULL_KEY, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _RETRIEVAL_PLANNER_PROMPT_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=700,
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


def _bounded_float(value, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return default


def _route_complexity(question: str, plan: QueryPlan) -> float:
    q = question.lower()
    score = 0.0
    if plan.query_type in {"why_how", "comparison", "methodology"}:
        score += 0.35
    if plan.query_type == "numerical" and plan.needs_calculation:
        score += 0.35
    if plan.evidence_need != "single_chunk":
        score += 0.25
    if plan.cross_table:
        score += 0.25
    if len(q.split()) >= 14:
        score += 0.15
    if re.search(r"(?i)\b(and|or|both|across|trade[- ]?off|why|how|compare|difference)\b", q):
        score += 0.15
    return min(1.0, score)


def _route_confidence(question: str, plan: QueryPlan) -> float:
    q = question.lower()
    signals = 0
    if plan.query_type == "numerical" and (_TABLE_RE.search(question) or _NUMERICAL_REASONING_RE.search(question)):
        signals += 2
    elif plan.query_type == "comparison" and _CROSS_TABLE_FALLBACK_RE.search(question):
        signals += 2
    elif plan.query_type == "why_how" and _SYNTHESIS_RE.search(question):
        signals += 2
    elif plan.query_type == "methodology" and re.search(r"(?i)\b(architect|train|loss|encoder|decoder|attention|method)\b", q):
        signals += 2
    elif plan.query_type == "factual":
        signals += 1
    if plan.is_yes_no:
        signals += 1
    return min(0.95, 0.55 + 0.15 * signals)


def _retrieval_strategy(plan: QueryPlan) -> str:
    if plan.evidence_need in {"two_tables", "two_sided_evidence"}:
        return "decompose"
    if plan.query_type in {"why_how", "methodology", "comparison", "numerical"}:
        return "fusion"
    return "standard"


def _attach_route_metadata(plan: QueryPlan, question: str,
                           route_level: str, classifier: str | None = None,
                           confidence: float | None = None,
                           retrieval_strategy: str | None = None,
                           rewrite_queries: list[str] | None = None,
                           hyde_query: str | None = None) -> QueryPlan:
    complexity = _route_complexity(question, plan)
    return replace(
        plan,
        classifier=classifier or plan.classifier,
        route_level=route_level,
        route_confidence=_bounded_float(confidence, _route_confidence(question, plan)),
        complexity_score=complexity,
        retrieval_strategy=retrieval_strategy or _retrieval_strategy(plan),
        rewrite_queries=rewrite_queries if rewrite_queries is not None else plan.rewrite_queries,
        hyde_query=hyde_query if hyde_query is not None else plan.hyde_query,
    )


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
    retrieval_strategy = raw.get("retrieval_strategy")
    if retrieval_strategy not in _RETRIEVAL_STRATEGIES:
        retrieval_strategy = fallback.retrieval_strategy
    rewrite_queries = _as_str_list(raw.get("rewrite_queries"))[:4]
    hyde_query = raw.get("hyde_query") if isinstance(raw.get("hyde_query"), str) else ""

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
        route_level="large_planner",
        route_confidence=0.9,
        complexity_score=fallback.complexity_score,
        retrieval_strategy=retrieval_strategy,
        rewrite_queries=rewrite_queries,
        hyde_query=hyde_query.strip()[:800],
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


def _small_model_route_query(question: str, fallback: QueryPlan) -> QueryPlan:
    """Fast local router layer used before calling a large planner.

    The deterministic rules handle high-precision obvious cases. If the local
    MiniLM router model is available, it may override a factual rule result only
    when the semantic label is clearly separated from the runner-up label.
    """
    plan = _attach_route_metadata(
        fallback,
        question,
        route_level="small_router",
        classifier="small_router",
    )
    semantic = predict_query_type(question)
    if semantic is None:
        return plan

    semantic_label = semantic["label"]
    semantic_score = float(semantic["score"])
    semantic_margin = float(semantic["margin"])
    should_override = (
        plan.query_type == "factual"
        and semantic_label != "factual"
        and semantic_score >= 0.55
        and semantic_margin >= 0.15
    )
    if not should_override:
        return plan

    semantic_plan = _rule_route_query(question, llm_client=None)
    semantic_plan = replace(semantic_plan, query_type=semantic_label)
    semantic_plan = _normalize_plan_for_query_type(semantic_plan)
    return _attach_route_metadata(
        semantic_plan,
        question,
        route_level="small_router_model",
        classifier="regex_plus_local_model",
        confidence=min(0.95, max(plan.route_confidence, semantic_score)),
    )


def _normalize_plan_for_query_type(plan: QueryPlan) -> QueryPlan:
    if plan.query_type == "numerical":
        return replace(
            plan,
            evidence_need="table_or_numeric_sentence",
            source_preference=["table", "paragraph"],
            answer_style="numeric_with_units",
            confidence_checks=list(dict.fromkeys(plan.confidence_checks + [
                "must_find_numeric_value",
                "prefer_table_evidence",
            ])),
            needs_calculation=plan.needs_calculation,
        )
    if plan.query_type == "comparison":
        return replace(
            plan,
            evidence_need="two_sided_evidence",
            answer_style="structured_comparison",
            confidence_checks=list(dict.fromkeys(plan.confidence_checks + [
                "require_two_sided_evidence",
            ])),
        )
    if plan.query_type == "methodology":
        return replace(
            plan,
            evidence_need="method_section",
            answer_style="method_steps",
            confidence_checks=list(dict.fromkeys(plan.confidence_checks + [
                "prefer_method_sections",
            ])),
        )
    if plan.query_type == "why_how":
        return replace(
            plan,
            evidence_need="multi_chunk_synthesis",
            answer_style="explanatory_synthesis",
            confidence_checks=list(dict.fromkeys(plan.confidence_checks + [
                "synthesize_multiple_chunks",
            ])),
        )
    return plan


def route_query(question: str, llm_client=None) -> QueryPlan:
    """Return the full policy plan for this question.

    Uses an LLM intent classifier when credentials are configured, then falls
    back to deterministic rules on missing credentials, network errors, quota
    errors, or malformed model output.
    """
    rule_plan = _rule_route_query(question, llm_client=None)
    small_plan = _small_model_route_query(question, rule_plan)
    if llm_client is None:
        return small_plan

    import os
    global _FULL_KEY
    _FULL_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    key_hint = _FULL_KEY[:8]

    if small_plan.route_confidence >= 0.85 and small_plan.complexity_score < 0.45:
        return small_plan

    try:
        raw = _cached_llm_retrieval_plan(question, model, base_url, key_hint)
        plan = _coerce_llm_plan(raw, small_plan)
        return _attach_route_metadata(
            plan,
            question,
            route_level="large_planner",
            classifier="llm_planner",
            confidence=0.9,
            retrieval_strategy=plan.retrieval_strategy,
            rewrite_queries=plan.rewrite_queries,
            hyde_query=plan.hyde_query,
        )
    except Exception:
        return small_plan
