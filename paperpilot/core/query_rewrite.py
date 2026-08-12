"""Query expansion policies shared by app and benchmark retrieval."""
from __future__ import annotations


def query_variants(query_text: str, query_plan: dict | None = None) -> list[str]:
    """Return intent-aware search views for scientific-paper retrieval.

    The original question is always first. Simple factual questions are left
    untouched unless an upstream planner explicitly provides rewrites.
    """
    query_plan = query_plan or {}
    query_type = query_plan.get("query_type", "factual")
    lowered = query_text.lower()
    synthesis_like = any(
        token in lowered
        for token in (
            "summarize", "summary", "core contribution", "main contribution",
            "contribution", "main idea", "overall", "limitation", "limitations",
            "motivation", "why it works",
        )
    )
    variants = [query_text]

    for rewrite in query_plan.get("rewrite_queries") or []:
        if isinstance(rewrite, str) and rewrite.strip():
            variants.append(rewrite.strip())

    hyde_query = query_plan.get("hyde_query")
    if isinstance(hyde_query, str) and hyde_query.strip():
        variants.append(hyde_query.strip())

    if len(variants) == 1:
        if query_type == "numerical":
            variants.extend([
                f"{query_text} table results metrics score accuracy f1 percentage number value",
                f"{query_text} highest lowest comparison result dataset model",
            ])
        elif query_type == "comparison":
            entities = " ".join(query_plan.get("comparison_entities") or [])
            variants.extend([
                f"{query_text} {entities} compare versus baseline difference performance results",
                f"{query_text} advantage disadvantage improvement ablation evaluation",
            ])
        elif query_type == "why_how" or synthesis_like:
            variants.extend([
                f"{query_text} motivation reason explanation contribution limitation evidence",
                f"{query_text} method result analysis discussion finding",
                "proposed method main contribution abstract introduction conclusion",
                "this work proposes model architecture self-attention results",
            ])
        elif query_type == "methodology":
            variants.extend([
                f"{query_text} method architecture training objective loss implementation",
                f"{query_text} model encoder decoder layer attention algorithm",
            ])

    deduped = []
    seen = set()
    for variant in variants:
        norm = " ".join(variant.split()).lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(variant)
    return deduped
