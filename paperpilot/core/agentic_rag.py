"""Tool-calling agentic RAG loop.

This module is intentionally separate from the app and benchmark paths. It
implements the core agent contract first: the model decides which tool to call,
observes the result, then decides whether to continue or stop.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from paperpilot.core.generator import generate_answer
from paperpilot.core.numeric_reasoner import numeric_reasoning_notes
from paperpilot.core.query_rewrite import query_variants


RetrieverFn = Callable[[str, int], list[dict]]
NeighborFn = Callable[[str, int], list[dict]]
ToolHandler = Callable[["AgentState", dict], dict]

_MAX_EVIDENCE = 16


@dataclass
class AgentStep:
    name: str
    observation: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "observation": self.observation,
            "data": self.data,
        }


@dataclass
class AgentState:
    question: str
    query_plan: dict
    evidence: list[dict] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    neighbor_expansions: int = 0
    neighbor_candidate_ids: set[str] = field(default_factory=set)
    rewrite_history: list[dict] = field(default_factory=list)
    decomposition_history: list[dict] = field(default_factory=list)
    last_verification: dict | None = None
    verified_answer_candidate: str = ""
    conservative_answer_candidate: str = ""
    final_answer_override: str = ""
    verification_attempts: int = 0
    failed_verifications: int = 0
    stop_requested: bool = False
    stop_reason: str = ""

    def add_step(self, name: str, observation: str, data: dict | None = None) -> None:
        self.steps.append(AgentStep(name=name, observation=observation, data=data or {}))

    def add_evidence(self, rows: list[dict], *, prepend: bool = False) -> list[str]:
        seen = {
            row.get("chunk", {}).get("chunk_id")
            for row in self.evidence
            if row.get("chunk", {}).get("chunk_id")
        }
        accepted: list[dict] = []
        accepted_ids: list[str] = []
        for row in rows:
            cid = row.get("chunk", {}).get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            accepted.append(row)
            accepted_ids.append(cid)

        if prepend:
            self.evidence = (accepted + self.evidence)[:_MAX_EVIDENCE]
        else:
            self.evidence = (self.evidence + accepted)[:_MAX_EVIDENCE]
        return accepted_ids

    def add_neighbor_evidence(self, anchor_id: str, rows: list[dict]) -> list[str]:
        """Insert neighboring chunks after their anchor when possible."""
        seen = {
            row.get("chunk", {}).get("chunk_id")
            for row in self.evidence
            if row.get("chunk", {}).get("chunk_id")
        }
        accepted: list[dict] = []
        accepted_ids: list[str] = []
        for row in rows:
            cid = row.get("chunk", {}).get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            accepted.append(row)
            accepted_ids.append(cid)

        if not accepted:
            return []

        insert_at = len(self.evidence)
        for idx, row in enumerate(self.evidence):
            if row.get("chunk", {}).get("chunk_id") == anchor_id:
                insert_at = idx + 1
                break
        self.evidence = (
            self.evidence[:insert_at] + accepted + self.evidence[insert_at:]
        )[:_MAX_EVIDENCE]
        return accepted_ids


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict
    handler: ToolHandler

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[AgentTool]):
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return [tool.schema() for tool in self._tools.values()]
        return [self._tools[name].schema() for name in names if name in self._tools]

    def run(self, name: str, state: AgentState, args: dict) -> dict:
        if name not in self._tools:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        return self._tools[name].handler(state, args)

    def names(self) -> list[str]:
        return list(self._tools)


def _object_schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _chunk_summary(row: dict, rank: int) -> dict:
    chunk = row.get("chunk", {})
    return {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id", ""),
        "chunk_type": chunk.get("chunk_type", "paragraph"),
        "section": chunk.get("section", ""),
        "score": round(float(row.get("score", 0.0)), 4),
        "preview": " ".join(chunk.get("text", "").split())[:280],
    }


def _dedupe_groups(groups: list[list[dict]]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            cid = row.get("chunk", {}).get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            rows.append(row)
    return rows


def _has_number(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _question_profile(question: str, query_plan: dict) -> dict:
    """Compact, visible task profile used to avoid one fixed tool sequence."""
    lowered = question.lower()
    query_type = query_plan.get("query_type", "factual")
    is_numeric = query_type == "numerical" or any(
        token in lowered
        for token in (
            "how many", "how much", "score", "accuracy", "f1", "percentage",
            "percent", "number", "rate", "higher", "highest", "lower", "lowest",
            "experimental result", "experimental results", "key results",
            "main results", "performance", "bleu", "rouge", "precision", "recall",
        )
    )
    is_comparison = query_type == "comparison" or any(
        token in lowered
        for token in ("compare", "difference", "different", "versus", " vs ", "which method")
    )
    is_method = query_type in {"why_how", "methodology"} or any(
        token in lowered
        for token in ("why", "how do", "how does", "method", "approach", "architecture")
    )
    is_synthesis = query_type == "why_how" or any(
        token in lowered
        for token in (
            "summarize", "summary", "core contribution", "main contribution",
            "contribution", "main idea", "overall", "limitation", "limitations",
            "motivation", "why it works",
        )
    )
    return {
        "query_type": query_type,
        "numeric_or_table": is_numeric,
        "comparison": is_comparison,
        "method_or_explanation": is_method or is_synthesis,
        "synthesis_or_contribution": is_synthesis,
        "suggested_start": (
            "retrieve_numeric_or_table"
            if is_numeric else "retrieve_fusion"
            if is_comparison or is_method or is_synthesis else "retrieve_evidence"
        ),
        "stop_test": (
            "Can the current evidence directly answer the question with specific source chunks?"
        ),
    }


def _evidence_status(state: AgentState) -> dict:
    sections = sorted({
        row.get("chunk", {}).get("section", "")
        for row in state.evidence
        if row.get("chunk", {}).get("section")
    })
    table_ids = [
        row.get("chunk", {}).get("chunk_id")
        for row in state.evidence
        if row.get("chunk", {}).get("chunk_type") in {"table", "merged_tables"}
    ]
    numeric_ids = [
        row.get("chunk", {}).get("chunk_id")
        for row in state.evidence
        if _has_number(row.get("chunk", {}).get("text", ""))
    ]
    return {
        "evidence_count": len(state.evidence),
        "chunk_ids": [
            row.get("chunk", {}).get("chunk_id") for row in state.evidence
        ],
        "sections": sections[:8],
        "table_chunk_ids": table_ids,
        "numeric_chunk_ids": numeric_ids[:8],
        "neighbor_expansions_used": state.neighbor_expansions,
        "neighbor_candidate_ids": sorted(state.neighbor_candidate_ids),
    }


def _suggest_next_actions(state: AgentState, *, last_added: list[str] | None = None,
                          retrieved: list[dict] | None = None) -> list[str]:
    profile = _question_profile(state.question, state.query_plan)
    status = _evidence_status(state)
    suggestions: list[str] = []
    retrieved = retrieved or []
    added = last_added or []

    if status["evidence_count"] == 0:
        suggestions.append(f"start_with_{profile['suggested_start']}")
        return suggestions

    if profile["numeric_or_table"] and not status["table_chunk_ids"] and not status["numeric_chunk_ids"]:
        suggestions.append("try_retrieve_numeric_or_table_with_metric_terms")
    if profile["comparison"] and len(status["sections"]) < 2:
        suggestions.append("try_retrieve_fusion_for_broader_coverage")

    top_texts = [
        row.get("chunk", {}).get("text", "") for row in retrieved[:3]
    ]
    boundary_markers = ("following", "below", "above", "previous", "next", "these", "this section")
    if (
        added
        and state.neighbor_expansions < 2
        and any(marker in text.lower() for text in top_texts for marker in boundary_markers)
    ):
        suggestions.append("consider_list_neighbor_chunks_for_incomplete_local_context")

    if not suggestions:
        suggestions.append("finish_if_answer_is_directly_supported_otherwise_rewrite_query")
    return suggestions


def _update_neighbor_candidates(state: AgentState, rows: list[dict]) -> None:
    boundary_markers = ("following", "below", "above", "previous", "next", "these", "this section")
    for row in rows[:5]:
        chunk = row.get("chunk", {})
        cid = chunk.get("chunk_id", "")
        text = chunk.get("text", "").lower()
        if cid and any(marker in text for marker in boundary_markers):
            state.neighbor_candidate_ids.add(cid)


def _decompose_queries(question: str, query_plan: dict) -> list[str]:
    variants = query_variants(question, query_plan)
    pieces: list[str] = []
    entities = [
        str(entity).strip()
        for entity in query_plan.get("comparison_entities") or []
        if str(entity).strip()
    ]
    if entities:
        pieces.extend(entities[:4])

    split_parts = re.split(
        r"(?i)\b(?: and | or | versus | vs\.? | compared with | compared to | difference between )\b",
        question,
    )
    for part in split_parts:
        part = re.sub(r"^[^\w]+|[^\w]+$", "", part).strip()
        if 3 <= len(part) <= 120:
            pieces.append(part)

    if query_plan.get("query_type") == "numerical":
        pieces.append(f"{question} exact number")
        pieces.append(f"{question} table")
    elif query_plan.get("query_type") in {"comparison", "why_how", "methodology"}:
        pieces.extend(variants[1:4])

    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        norm = " ".join(piece.split()).lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(piece)
    return out[:6]


def _answer_claims(answer: str) -> list[str]:
    answer = re.sub(r"\[[0-9,\s]+\]", "", answer or "")
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", answer)
    claims = []
    for part in parts:
        claim = " ".join(part.split()).strip(" -")
        if len(claim) >= 12:
            claims.append(claim)
    return claims[:12]


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with",
        "that", "this", "is", "are", "was", "were", "be", "by", "as", "from",
        "it", "they", "their", "them", "which", "what", "how", "does", "did",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        if token not in stop
    }


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?", text or "")


def _verify_claim_against_evidence(claim: str, evidence: list[dict]) -> dict:
    claim_tokens = _content_tokens(claim)
    claim_numbers = _numbers(claim)
    best = {
        "chunk_id": "",
        "overlap": 0,
        "token_support": 0.0,
        "number_support": 1.0 if not claim_numbers else 0.0,
        "preview": "",
    }
    for row in evidence:
        chunk = row.get("chunk", {})
        text = chunk.get("text", "")
        text_lower = text.lower()
        overlap = len(claim_tokens & _content_tokens(text))
        token_support = overlap / max(1, len(claim_tokens))
        number_support = (
            1.0
            if not claim_numbers
            else sum(1 for num in claim_numbers if num.lower() in text_lower) / len(claim_numbers)
        )
        score = (token_support * 0.75) + (number_support * 0.25)
        best_score = (best["token_support"] * 0.75) + (best["number_support"] * 0.25)
        if score > best_score:
            best = {
                "chunk_id": chunk.get("chunk_id", ""),
                "overlap": overlap,
                "token_support": round(token_support, 3),
                "number_support": round(number_support, 3),
                "preview": " ".join(text.split())[:220],
            }
    has_numbers = bool(claim_numbers)
    supported = (
        best["number_support"] >= 1.0
        and (
            best["token_support"] >= 0.35
            or (not has_numbers and best["overlap"] >= 3)
        )
    )
    return {
        "claim": claim,
        "supported": supported,
        "best_chunk_id": best["chunk_id"],
        "token_support": best["token_support"],
        "number_support": best["number_support"],
        "missing_numbers": [
            num for num in claim_numbers
            if best["number_support"] < 1.0
        ],
        "evidence_preview": best["preview"],
    }


def _available_tool_names(state: AgentState, registry: ToolRegistry,
                          used_retrieval: bool) -> list[str]:
    """Expose tools dynamically based on observation state."""
    names = set(registry.names())
    if state.failed_verifications >= 2 and state.evidence:
        return [name for name in ["finish"] if name in names]

    available = [
        "rewrite_query",
        "decompose_question",
        "retrieve_evidence",
        "retrieve_fusion",
        "retrieve_numeric_or_table",
    ]
    status = _evidence_status(state)
    if used_retrieval and status["evidence_count"] > 0:
        if state.failed_verifications >= 1:
            return [
                name for name in [
                    "retrieve_evidence",
                    "retrieve_fusion",
                    "retrieve_numeric_or_table",
                    "verify_answer_support",
                    "finish",
                ]
                if name in names
            ]
        if (
            "list_neighbor_chunks" in names
            and state.neighbor_expansions < 2
            and state.neighbor_candidate_ids
        ):
            available.append("list_neighbor_chunks")
        if status["table_chunk_ids"]:
            available.append("extract_table_data")
        available.append("inspect_context")
        available.append("verify_answer_support")
        available.append("finish")
    return [name for name in available if name in names]


def _normalize_table(chunk: dict) -> dict:
    columns = chunk.get("table_columns") or []
    rows = chunk.get("table_rows") or []
    if columns and rows:
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "caption": chunk.get("caption", chunk.get("section", "")),
            "columns": columns,
            "rows": rows[:20],
        }

    markdown_rows = []
    for line in chunk.get("text", "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells if cell):
            continue
        markdown_rows.append(cells)

    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "caption": chunk.get("caption", chunk.get("section", "")),
        "columns": markdown_rows[0] if markdown_rows else [],
        "rows": markdown_rows[1:21] if len(markdown_rows) > 1 else [],
    }


def build_agentic_tools(retrieve_fn: RetrieverFn, query_plan: dict | None = None,
                        neighbor_fn: NeighborFn | None = None) -> ToolRegistry:
    """Build local tools available to the model-controlled loop.

    The table tool is deliberately exposed as a normal agent tool. If a future
    mature parser is added as a skill/plugin, it can replace this handler while
    preserving the same tool contract.
    """
    query_plan = query_plan or {}

    def rewrite_query(state: AgentState, args: dict) -> dict:
        focus = str(args.get("focus") or "").strip()
        source_query = focus or state.question
        variants = query_variants(source_query, state.query_plan or query_plan)
        decomposition = _decompose_queries(state.question, state.query_plan or query_plan)
        profile = _question_profile(state.question, state.query_plan or query_plan)
        recommended = profile["suggested_start"]
        if state.evidence:
            recommended = (
                "retrieve_numeric_or_table"
                if profile["numeric_or_table"] else "retrieve_fusion"
            )
        output = {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "source_query": source_query,
            "rewrites": variants,
            "hyde_query": (state.query_plan or query_plan).get("hyde_query", ""),
            "decomposed_queries": decomposition,
            "recommended_retrieval_tool": recommended,
            "why_these_rewrites": (
                "Rewrites separate lexical variants from retrieval so the next "
                "agent turn can choose a retrieval tool after observing them."
            ),
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": [
                f"call_{recommended}_with_the_best_rewrite",
                "call_decompose_question_if_entities_or_subquestions_are_unclear",
            ],
        }
        state.rewrite_history.append(output)
        return output

    def decompose_question(state: AgentState, args: dict) -> dict:
        subqueries = _decompose_queries(state.question, state.query_plan or query_plan)
        output = {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "subqueries": subqueries,
            "recommended_use": (
                "Search each subquery only if current evidence misses that part; "
                "do not retrieve every subquery mechanically."
            ),
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": [
                "retrieve_missing_subquestion",
                "finish_if_existing_evidence_covers_all_subqueries",
            ],
        }
        state.decomposition_history.append(output)
        return output

    def retrieve_evidence(state: AgentState, args: dict) -> dict:
        query = str(args.get("query") or state.question)
        top_k = int(args.get("top_k") or 5)
        rows = retrieve_fn(query, top_k)
        added = state.add_evidence(rows)
        _update_neighbor_candidates(state, rows)
        summaries = [_chunk_summary(row, idx + 1) for idx, row in enumerate(rows)]
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "added_chunk_ids": added,
            "retrieved": summaries,
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(
                state,
                last_added=added,
                retrieved=rows,
            ),
        }

    def retrieve_fusion(state: AgentState, args: dict) -> dict:
        query = str(args.get("query") or state.question)
        top_k = int(args.get("top_k") or 7)
        variants = query_variants(query, state.query_plan or query_plan)
        groups = [retrieve_fn(variant, max(top_k, 7)) for variant in variants]
        rows = _dedupe_groups(groups)[:top_k]
        added = state.add_evidence(rows)
        _update_neighbor_candidates(state, rows)
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "variants": variants,
            "added_chunk_ids": added,
            "retrieved": [_chunk_summary(row, idx + 1) for idx, row in enumerate(rows)],
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(
                state,
                last_added=added,
                retrieved=rows,
            ),
        }

    def retrieve_numeric_or_table(state: AgentState, args: dict) -> dict:
        query = str(args.get("query") or state.question)
        top_k = int(args.get("top_k") or 7)
        expanded = f"{query} table results metrics score accuracy f1 number value"
        rows = retrieve_fn(expanded, max(top_k, 7))
        table_or_numeric = [
            row for row in rows
            if row.get("chunk", {}).get("chunk_type") in {"table", "merged_tables"}
            or _has_number(row.get("chunk", {}).get("text", ""))
        ]
        selected = _dedupe_groups([table_or_numeric, rows])[:top_k]
        has_real_table = any(
            row.get("chunk", {}).get("chunk_type") in {"table", "merged_tables"}
            for row in selected
        )
        added = state.add_evidence(selected, prepend=has_real_table)
        _update_neighbor_candidates(state, selected)
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "expanded_query": expanded,
            "prepend_policy": "prepend_real_table_only" if has_real_table else "append_no_table_found",
            "added_chunk_ids": added,
            "retrieved": [_chunk_summary(row, idx + 1) for idx, row in enumerate(selected)],
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(
                state,
                last_added=added,
                retrieved=selected,
            ),
        }

    def list_neighbor_chunks(state: AgentState, args: dict) -> dict:
        if neighbor_fn is None:
            return {
                "ok": False,
                "error": "Neighbor chunk listing is not available in this runtime.",
            }
        if state.neighbor_expansions >= 2:
            return {
                "ok": True,
                "agent_reason": str(args.get("reason") or ""),
                "anchor_chunk_id": str(args.get("chunk_id") or ""),
                "window": 0,
                "added_chunk_ids": [],
                "neighbors": [],
                "note": "Neighbor expansion budget exhausted for this question.",
                "evidence_status": _evidence_status(state),
                "suggested_next_actions": _suggest_next_actions(state),
            }
        chunk_id = str(args.get("chunk_id") or "")
        window = 1
        if not chunk_id:
            return {"ok": False, "error": "chunk_id is required"}
        rows = neighbor_fn(chunk_id, window)
        added = state.add_neighbor_evidence(chunk_id, rows)
        state.neighbor_candidate_ids.discard(chunk_id)
        state.neighbor_expansions += 1
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "anchor_chunk_id": chunk_id,
            "window": window,
            "added_chunk_ids": added,
            "neighbors": [_chunk_summary(row, idx + 1) for idx, row in enumerate(rows)],
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(
                state,
                last_added=added,
                retrieved=rows,
            ),
        }

    def extract_table_data(state: AgentState, args: dict) -> dict:
        chunk_id = str(args.get("chunk_id") or "")
        table_chunks = [
            row.get("chunk", {}) for row in state.evidence
            if row.get("chunk", {}).get("chunk_type") in {"table", "merged_tables"}
        ]
        if chunk_id:
            table_chunks = [chunk for chunk in table_chunks if chunk.get("chunk_id") == chunk_id]
        if not table_chunks:
            return {
                "ok": False,
                "error": (
                    "No table evidence is loaded. Do not call extract_table_data "
                    "unless evidence_status.table_chunk_ids is non-empty."
                ),
                "agent_reason": str(args.get("reason") or ""),
                "tables_found": 0,
                "tables": [],
                "evidence_status": _evidence_status(state),
                "suggested_next_actions": [
                    "finish_if_paragraph_evidence_answers_the_question",
                    "otherwise_call_retrieve_numeric_or_table",
                ],
            }
        tables = [_normalize_table(chunk) for chunk in table_chunks]
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "tables_found": len(tables),
            "tables": tables,
            "note": "Structured table rows extracted from current evidence.",
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(state),
        }

    def inspect_context(state: AgentState, _args: dict) -> dict:
        notes, facts = numeric_reasoning_notes(state.question, state.evidence, state.query_plan)
        by_chunk: dict[str, list[dict]] = {}
        for fact in facts:
            by_chunk.setdefault(fact.get("chunk_id", ""), []).append(fact)
        for row in state.evidence:
            cid = row.get("chunk", {}).get("chunk_id", "")
            row["numeric_facts"] = by_chunk.get(cid, [])[:12]

        sections = {
            row.get("chunk", {}).get("section", "")
            for row in state.evidence
            if row.get("chunk", {}).get("section")
        }
        table_count = sum(
            1 for row in state.evidence
            if row.get("chunk", {}).get("chunk_type") in {"table", "merged_tables"}
        )
        query_type = state.query_plan.get("query_type", "factual")
        enough = bool(state.evidence)
        gaps: list[str] = []
        if not state.evidence:
            gaps.append("no evidence retrieved")
        if query_type == "numerical" and not facts and table_count == 0:
            enough = False
            gaps.append("numeric/table evidence missing")
        if query_type in {"comparison", "why_how", "methodology"} and len(sections) < 2:
            enough = False
            gaps.append("multi-source evidence may be insufficient")

        return {
            "ok": True,
            "agent_reason": str(_args.get("reason") or "") if isinstance(_args, dict) else "",
            "sufficient": enough,
            "gaps": gaps,
            "evidence_count": len(state.evidence),
            "unique_sections": len(sections),
            "table_count": table_count,
            "numeric_fact_count": len(facts),
            "numeric_notes": notes,
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": _suggest_next_actions(state),
        }

    def finish(state: AgentState, args: dict) -> dict:
        state.stop_requested = True
        state.stop_reason = str(args.get("reason") or "model decided evidence is sufficient")
        final_answer = str(args.get("final_answer") or "").strip()
        if final_answer:
            state.final_answer_override = final_answer
        return {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "stop": True,
            "reason": state.stop_reason,
            "final_answer": state.final_answer_override,
            "final_evidence_count": len(state.evidence),
            "evidence_status": _evidence_status(state),
            "last_verification": state.last_verification,
        }

    def verify_answer_support(state: AgentState, args: dict) -> dict:
        state.verification_attempts += 1
        candidate = str(args.get("candidate_answer") or "").strip()
        if not candidate:
            state.failed_verifications += 1
            output = {
                "ok": False,
                "error": "candidate_answer is required so claims can be checked against current evidence.",
                "agent_reason": str(args.get("reason") or ""),
                "evidence_status": _evidence_status(state),
                "suggested_next_actions": ["draft_candidate_answer_from_current_evidence_then_verify_again"],
            }
            state.last_verification = output
            return output
        claims = _answer_claims(candidate)
        if not claims:
            state.failed_verifications += 1
            output = {
                "ok": False,
                "error": "candidate_answer did not contain checkable factual claims.",
                "agent_reason": str(args.get("reason") or ""),
                "evidence_status": _evidence_status(state),
                "suggested_next_actions": ["rewrite_candidate_answer_as_specific_evidence_based_claims"],
            }
            state.last_verification = output
            return output
        checked = [_verify_claim_against_evidence(claim, state.evidence) for claim in claims]
        supported = [item for item in checked if item["supported"]]
        unsupported = [item for item in checked if not item["supported"]]
        citation_ids = set(re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", candidate))
        coverage = len(supported) / max(1, len(checked))
        supported_answer_candidate = " ".join(
            item["claim"].rstrip(".") + "." for item in supported
        )
        verified = coverage >= 0.8 and not unsupported
        if verified:
            state.failed_verifications = 0
        else:
            state.failed_verifications += 1
            if supported_answer_candidate:
                state.conservative_answer_candidate = supported_answer_candidate
        if state.failed_verifications >= 2:
            next_actions = ["finish_with_supported_answer_candidate_or_state_insufficient_evidence"]
        elif verified:
            next_actions = ["finish_with_verified_answer"]
        else:
            next_actions = [
                "one_more_targeted_retrieval_for_unsupported_claims",
                "or_finish_with_supported_answer_candidate_if_it_answers_the_question",
            ]
        output = {
            "ok": True,
            "agent_reason": str(args.get("reason") or ""),
            "candidate_answer": candidate,
            "claims_checked": checked,
            "supported_claim_count": len(supported),
            "unsupported_claim_count": len(unsupported),
            "supported_answer_candidate": supported_answer_candidate,
            "support_coverage": round(coverage, 3),
            "citation_mentions": sorted(citation_ids),
            "verified": verified,
            "verification_attempts": state.verification_attempts,
            "failed_verifications": state.failed_verifications,
            "evidence_status": _evidence_status(state),
            "suggested_next_actions": next_actions,
        }
        if output["verified"]:
            state.verified_answer_candidate = candidate
        state.last_verification = output
        return output

    tools = [
        AgentTool(
            "rewrite_query",
            "Generate search rewrites and a recommended retrieval tool without retrieving evidence.",
            _object_schema({
                "reason": {"type": "string"},
                "focus": {"type": "string"},
            }, ["reason"]),
            rewrite_query,
        ),
        AgentTool(
            "decompose_question",
            "Break a complex, numeric, or comparison question into focused subqueries without retrieving evidence.",
            _object_schema({
                "reason": {"type": "string"},
            }, ["reason"]),
            decompose_question,
        ),
        AgentTool(
            "retrieve_evidence",
            "Retrieve relevant paper chunks for a direct factual search query.",
            _object_schema({
                "reason": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
            }, ["reason", "query"]),
            retrieve_evidence,
        ),
        AgentTool(
            "retrieve_fusion",
            "Run multiple query rewrites and merge their evidence; use for comparison, method, why/how, or failed narrow retrieval.",
            _object_schema({
                "reason": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
            }, ["reason", "query"]),
            retrieve_fusion,
        ),
        AgentTool(
            "retrieve_numeric_or_table",
            "Retrieve evidence biased toward tables, metrics, and numeric values; use only when the question or observation needs numbers/tables.",
            _object_schema({
                "reason": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
            }, ["reason", "query"]),
            retrieve_numeric_or_table,
        ),
    ]
    if neighbor_fn is not None:
        tools.append(AgentTool(
            "list_neighbor_chunks",
            "List chunks immediately before and after one retrieved chunk from the same paper; use only when the observed chunk is locally incomplete.",
            _object_schema({
                "reason": {"type": "string"},
                "chunk_id": {"type": "string"},
                "window": {"type": "integer", "minimum": 1, "maximum": 3},
            }, ["reason", "chunk_id"]),
            list_neighbor_chunks,
        ))
    tools.extend([
        AgentTool(
            "extract_table_data",
            "Extract structured rows from currently loaded table evidence.",
            _object_schema({
                "reason": {"type": "string"},
                "chunk_id": {"type": "string"},
            }, ["reason"]),
            extract_table_data,
        ),
        AgentTool(
            "inspect_context",
            "Inspect whether the current evidence appears sufficient and identify gaps; use when the latest observation is ambiguous.",
            _object_schema({
                "reason": {"type": "string"},
            }, ["reason"]),
            inspect_context,
        ),
        AgentTool(
            "finish",
            "Stop the loop when enough evidence has been collected.",
            _object_schema({
                "reason": {"type": "string"},
                "final_answer": {"type": "string"},
            }, ["reason"]),
            finish,
        ),
        AgentTool(
            "verify_answer_support",
            "Check whether a candidate answer's factual claims are grounded in the currently loaded evidence.",
            _object_schema({
                "reason": {"type": "string"},
                "candidate_answer": {"type": "string"},
            }, ["reason", "candidate_answer"]),
            verify_answer_support,
        ),
    ])
    return ToolRegistry(tools)


_SYSTEM_PROMPT = """You are PaperPilot operating in Work Mode.
You must solve the user's paper question as an observation-driven tool agent.

Rules:
- Call exactly one tool per turn, then wait for its observation before deciding the next action.
- Do not follow a fixed retrieve -> neighbor -> inspect -> finish workflow.
- Every tool call must include a concise reason grounded in the current question or the latest tool observation.
- Use rewrite_query or decompose_question at most once, only when the first search wording is uncertain or retrieval came back weak.
- Use at least one retrieval tool before finishing.
- For simple factual questions, use retrieve_evidence and finish as soon as evidence directly answers the question.
- For summary, contribution, motivation, limitation, or "what is this paper about" questions, use retrieve_fusion for broad coverage before finishing.
- For contribution questions, distinguish the paper's proposed method from prior work or background limitations.
- For tables, metrics, counts, or scores, use retrieve_numeric_or_table; call extract_table_data only after a table chunk is actually present.
- For comparison, why/how, methodology, or broad coverage questions, use retrieve_fusion.
- Call list_neighbor_chunks only when an observed chunk is locally incomplete, references content before/after it, or the answer likely spans adjacent chunks. Do not call it just because the tool exists.
- Call inspect_context only when the latest observation leaves uncertainty about sufficiency or gaps.
- Use verify_answer_support before finish for numeric answers, comparisons, or answers with multiple factual claims. Verify at most twice.
- If verification fails once, do one targeted retrieval or finish with the supported answer candidate.
- If verification fails twice, call finish with a conservative final_answer using only supported claims, or say evidence is insufficient.
- Stop by calling finish when evidence is sufficient or further retrieval is unlikely to help.
- Keep tool queries short and targeted. Do not invent evidence."""


def _message_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tool_call_parts(call: Any) -> tuple[str, str, dict]:
    call_id = _message_attr(call, "id", "")
    function = _message_attr(call, "function", {})
    name = _message_attr(function, "name", "")
    raw_args = _message_attr(function, "arguments", "{}") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except Exception:
        args = {}
    return call_id, name, args


def _assistant_message(message: Any) -> dict:
    tool_calls = _message_attr(message, "tool_calls", None) or []
    converted = []
    for call in tool_calls:
        call_id, name, args = _tool_call_parts(call)
        converted.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return {
        "role": "assistant",
        "content": _message_attr(message, "content", None),
        "tool_calls": converted,
    }


def _redact_message_for_trace(message: dict) -> dict:
    """Return the complete visible protocol message for audit storage.

    This records user/system prompts, assistant visible content, tool calls, and
    tool observations. It does not expose hidden model chain-of-thought because
    that is not available through the API and should not be treated as product
    telemetry.
    """
    return json.loads(json.dumps(message, ensure_ascii=False))


def run_agentic_rag(question: str, query_plan: dict, retrieve_fn: RetrieverFn,
                    llm_client: Any, top_k: int = 5,
                    neighbor_fn: NeighborFn | None = None,
                    max_steps: int = 7, model: str | None = None) -> dict:
    """Run a real tool-calling RAG loop.

    Args:
        question: User question.
        query_plan: Router output used as context, not as a fixed plan.
        retrieve_fn: Local retrieval function `(query, top_k) -> retrieved rows`.
        llm_client: OpenAI-compatible client that supports chat.completions tools.
        top_k: Number of final evidence chunks passed to answer generation.
        max_steps: Hard guardrail, similar to Work Mode auto-pause.
        model: Optional model override.
    """
    if llm_client is None:
        raise ValueError("run_agentic_rag requires a tool-calling LLM client")

    state = AgentState(question=question, query_plan=query_plan or {})
    registry = build_agentic_tools(retrieve_fn, query_plan, neighbor_fn=neighbor_fn)
    state.add_step(
        "tool_registry",
        f"Registered {len(registry.names())} tools.",
        {"tools": registry.names(), "max_steps": max_steps},
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "query_plan": query_plan or {},
                "question_profile": _question_profile(question, query_plan or {}),
                "instruction": (
                    "Choose the next single tool from the current observation. "
                    "Different question types should use different tool paths."
                ),
            }, ensure_ascii=False),
        },
    ]
    used_tool = False

    for step_no in range(1, max_steps + 1):
        available_tools = _available_tool_names(state, registry, used_tool)
        visible_messages_before = [_redact_message_for_trace(message) for message in messages]
        create_kwargs = {
            "model": model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": messages,
            "tools": registry.schemas(available_tools),
            "tool_choice": "auto",
            "temperature": 0,
            "parallel_tool_calls": False,
        }
        try:
            response = llm_client.chat.completions.create(**create_kwargs)
        except TypeError:
            create_kwargs.pop("parallel_tool_calls", None)
            response = llm_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        tool_calls = _message_attr(message, "tool_calls", None) or []
        round_record = {
            "round": step_no,
            "messages_before": visible_messages_before,
            "assistant_message": _assistant_message(message),
            "tool_results": [],
            "available_tools": available_tools,
            "evidence_chunk_ids_before": [
                row.get("chunk", {}).get("chunk_id") for row in state.evidence
            ],
        }

        if not tool_calls:
            content = (_message_attr(message, "content", "") or "").strip()
            state.add_step(
                "model_stop",
                "Model returned a final message without tool calls.",
                {"content": content, "step": step_no},
            )
            round_record["evidence_chunk_ids_after"] = [
                row.get("chunk", {}).get("chunk_id") for row in state.evidence
            ]
            state.rounds.append(round_record)
            break

        messages.append(_assistant_message(message))
        for call_idx, call in enumerate(tool_calls):
            call_id, name, args = _tool_call_parts(call)
            used_tool = True
            if call_idx == 0:
                output = registry.run(name, state, args)
            else:
                output = {
                    "ok": False,
                    "error": (
                        "Only one tool call is executed per turn. Observe the first "
                        "tool result before deciding another action."
                    ),
                    "evidence_status": _evidence_status(state),
                    "suggested_next_actions": _suggest_next_actions(state),
                }
            state.add_step(
                f"tool:{name}",
                _tool_observation(name, output),
                {"args": args, "output": output, "step": step_no},
            )
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(output, ensure_ascii=False),
            }
            messages.append(tool_message)
            round_record["tool_results"].append({
                "tool_call_id": call_id,
                "tool_name": name,
                "args": args,
                "output": output,
                "tool_message": _redact_message_for_trace(tool_message),
            })

        round_record["evidence_chunk_ids_after"] = [
            row.get("chunk", {}).get("chunk_id") for row in state.evidence
        ]
        state.rounds.append(round_record)
        if state.stop_requested:
            break

    else:
        state.add_step(
            "guardrail_pause",
            f"Stopped after max_steps={max_steps}.",
            {"max_steps": max_steps, "used_tool": used_tool},
        )

    final_evidence = state.evidence[:top_k]
    if final_evidence:
        answer = generate_answer(question, final_evidence, query_plan)
    elif state.final_answer_override:
        answer = state.final_answer_override
    elif state.verified_answer_candidate:
        answer = state.verified_answer_candidate
    elif state.conservative_answer_candidate:
        answer = state.conservative_answer_candidate
    else:
        answer = "Not enough information was retrieved to answer from the sources."

    state.add_step(
        "generate",
        "Generated final answer from the agent-selected evidence set.",
        {
            "final_chunk_ids": [
                row.get("chunk", {}).get("chunk_id") for row in final_evidence
            ],
            "stop_reason": state.stop_reason,
        },
    )

    return {
        "answer": answer,
        "retrieved": final_evidence,
        "trace": [step.to_dict() for step in state.steps],
        "rounds": state.rounds,
    }


def _tool_observation(name: str, output: dict) -> str:
    if not output.get("ok", False):
        return f"{name} failed: {output.get('error', 'unknown error')}"
    if name.startswith("retrieve"):
        added = output.get("added_chunk_ids", [])
        return f"{name} added {len(added)} new chunk(s)."
    if name == "inspect_context":
        return (
            "Context sufficient."
            if output.get("sufficient")
            else "Context inspection found evidence gaps."
        )
    if name == "extract_table_data":
        return f"Extracted {output.get('tables_found', 0)} table(s)."
    if name == "list_neighbor_chunks":
        return f"Listed neighbors and added {len(output.get('added_chunk_ids', []))} chunk(s)."
    if name == "finish":
        return f"Finish requested: {output.get('reason', '')}"
    if name == "rewrite_query":
        return f"Generated {len(output.get('rewrites', []))} rewrite(s)."
    if name == "decompose_question":
        return f"Generated {len(output.get('subqueries', []))} subquery/subqueries."
    if name == "verify_answer_support":
        return (
            "Candidate answer is supported."
            if output.get("verified")
            else f"Candidate support incomplete: {output.get('unsupported_claim_count', 0)} unsupported claim(s)."
        )
    return f"{name} completed."
