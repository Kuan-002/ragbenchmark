"""Render agentic QA JSONL traces into a human-readable Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _short(text: str, limit: int = 360) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _tool_summary(result: dict) -> list[str]:
    output = result.get("output") or {}
    lines = []
    tool_name = result.get("tool_name", "")
    args = result.get("args") or {}
    lines.append(f"- Tool: `{tool_name}`")
    if args:
        reason = args.get("reason")
        if reason:
            lines.append(f"  Reason: {_short(reason, 220)}")
        filtered_args = {key: value for key, value in args.items() if key != "reason"}
        if filtered_args:
            lines.append(f"  Args: `{json.dumps(filtered_args, ensure_ascii=False)}`")
    if not output.get("ok", False):
        lines.append(f"  Error: {output.get('error', 'unknown error')}")
        return lines

    if tool_name.startswith("retrieve"):
        added = output.get("added_chunk_ids", [])
        lines.append(f"  Added: {', '.join(added) if added else '(none)'}")
        if output.get("variants"):
            lines.append("  Query variants:")
            for variant in output["variants"]:
                lines.append(f"    - {_short(variant, 140)}")
        for item in output.get("retrieved", [])[:5]:
            lines.append(
                "  Retrieved "
                f"#{item.get('rank')} `{item.get('chunk_id')}` "
                f"score={item.get('score')} section={item.get('section')}: "
                f"{_short(item.get('preview'), 220)}"
            )
        if output.get("prepend_policy"):
            lines.append(f"  Policy: {output['prepend_policy']}")
    elif tool_name == "list_neighbor_chunks":
        lines.append(f"  Anchor: `{output.get('anchor_chunk_id', '')}` window={output.get('window')}")
        lines.append(f"  Added: {', '.join(output.get('added_chunk_ids', [])) or '(none)'}")
        if output.get("note"):
            lines.append(f"  Note: {output['note']}")
        for item in output.get("neighbors", [])[:6]:
            lines.append(
                "  Neighbor "
                f"`{item.get('chunk_id')}` section={item.get('section')}: "
                f"{_short(item.get('preview'), 220)}"
            )
    elif tool_name == "inspect_context":
        lines.append(
            "  Inspection: "
            f"sufficient={output.get('sufficient')} "
            f"gaps={output.get('gaps', [])} "
            f"evidence_count={output.get('evidence_count')} "
            f"sections={output.get('unique_sections')} "
            f"tables={output.get('table_count')} "
            f"numeric_facts={output.get('numeric_fact_count')}"
        )
        notes = output.get("numeric_notes")
        if notes and notes != "None.":
            lines.append(f"  Numeric notes: {_short(notes, 260)}")
    elif tool_name == "extract_table_data":
        lines.append(f"  Tables found: {output.get('tables_found')}")
        if output.get("note"):
            lines.append(f"  Note: {output['note']}")
    elif tool_name == "finish":
        lines.append(f"  Stop reason: {output.get('reason', '')}")
        verification = output.get("last_verification")
        if verification:
            lines.append(
                "  Last verification: "
                f"verified={verification.get('verified')} "
                f"coverage={verification.get('support_coverage')} "
                f"unsupported={verification.get('unsupported_claim_count')}"
            )
    elif tool_name == "rewrite_query":
        lines.append(f"  Recommended retrieval: `{output.get('recommended_retrieval_tool', '')}`")
        for rewrite in output.get("rewrites", [])[:6]:
            lines.append(f"  Rewrite: {_short(rewrite, 180)}")
        for subquery in output.get("decomposed_queries", [])[:6]:
            lines.append(f"  Decomposed: {_short(subquery, 180)}")
    elif tool_name == "decompose_question":
        for subquery in output.get("subqueries", [])[:8]:
            lines.append(f"  Subquery: {_short(subquery, 180)}")
        if output.get("recommended_use"):
            lines.append(f"  Use: {_short(output['recommended_use'], 240)}")
    elif tool_name == "verify_answer_support":
        lines.append(
            "  Verification: "
            f"verified={output.get('verified')} "
            f"coverage={output.get('support_coverage')} "
            f"supported={output.get('supported_claim_count')} "
            f"unsupported={output.get('unsupported_claim_count')}"
        )
        if output.get("supported_answer_candidate"):
            lines.append(
                f"  Supported answer candidate: {_short(output['supported_answer_candidate'], 360)}"
            )
        for claim in output.get("claims_checked", [])[:8]:
            status = "supported" if claim.get("supported") else "unsupported"
            lines.append(
                f"  Claim ({status}) `{claim.get('best_chunk_id', '')}` "
                f"token={claim.get('token_support')} number={claim.get('number_support')}: "
                f"{_short(claim.get('claim'), 220)}"
            )
    else:
        lines.append(f"  Output: `{json.dumps(output, ensure_ascii=False)[:500]}`")

    suggestions = output.get("suggested_next_actions") or []
    if suggestions:
        lines.append(f"  Suggested next: {', '.join(suggestions)}")
    status = output.get("evidence_status") or {}
    if status:
        lines.append(
            "  Evidence status: "
            f"count={status.get('evidence_count')} "
            f"tables={len(status.get('table_chunk_ids') or [])} "
            f"numeric={len(status.get('numeric_chunk_ids') or [])} "
            f"neighbors_used={status.get('neighbor_expansions_used')}"
        )
    return lines


def render(records: list[dict]) -> str:
    lines = ["# Agentic QA Trace Report", ""]
    for idx, rec in enumerate(records, 1):
        hit1 = "yes" if rec.get("hit1") else "no"
        hit5 = "yes" if rec.get("hit5") else "no"
        lines += [
            f"## {idx}. {rec.get('question', '')}",
            "",
            f"- QID: `{rec.get('qid', '')}`",
            f"- Reference answer: {_short(rec.get('reference_answer'), 420)}",
            f"- Gold ids: {', '.join(rec.get('gold_ids', []))}",
            f"- Retrieved ids: {', '.join(rec.get('retrieved_ids', []))}",
            f"- Hit@1: {hit1}; Hit@5: {hit5}; MRR: {rec.get('mrr'):.4f}; Recall@5: {rec.get('evidence_recall5'):.4f}",
            f"- Latency: {rec.get('latency_ms'):.1f} ms",
            "",
            "Final answer:",
            "",
            _short(rec.get("answer"), 900),
            "",
            "### Tool Rounds",
            "",
        ]
        for round_rec in rec.get("rounds", []):
            lines.append(f"#### Round {round_rec.get('round')}")
            assistant = round_rec.get("assistant_message", {})
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                content = assistant.get("content") or ""
                lines.append(f"Assistant final message: {_short(content, 400)}")
            else:
                call_names = [
                    call.get("function", {}).get("name", "")
                    for call in tool_calls
                ]
                lines.append(f"Assistant called: {', '.join(f'`{name}`' for name in call_names)}")
            before = round_rec.get("evidence_chunk_ids_before") or []
            after = round_rec.get("evidence_chunk_ids_after") or []
            available = round_rec.get("available_tools") or []
            if available:
                lines.append(f"Available tools: {', '.join(f'`{name}`' for name in available)}")
            lines.append(f"Evidence before: {', '.join(before) if before else '(none)'}")
            lines.append(f"Evidence after: {', '.join(after) if after else '(none)'}")
            for result in round_rec.get("tool_results", []):
                lines.extend(_tool_summary(result))
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Agentic trace JSONL path.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(records), encoding="utf-8")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
