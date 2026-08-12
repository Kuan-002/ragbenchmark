"""Route all QASPER questions, then evaluate against post-hoc proxy labels.

QASPER provides official answer-form labels, commonly treated as five classes
when yes/no is split: extractive, free_form, yes, no, unanswerable.

Those are not the same as this project's retrieval-intent router categories:
numerical, comparison, why_how, methodology, factual. Therefore router accuracy
below is measured against question-only proxy labels, while official answer-form
labels are preserved in the JSONL for auxiliary analysis.

To avoid information leakage:
1. route_query(question, llm_client=None) is called using ONLY the question.
2. by default, proxy labels are also question-only to avoid answer/evidence
   leakage. An optional answer-aware proxy exists for diagnostics only.

The reported score is proxy-label accuracy, not a human gold-label accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.dataset import load_qasper
from paperpilot.core.router import route_query


QUERY_TYPES = ["numerical", "comparison", "why_how", "methodology", "factual"]

NUMERIC_RE = re.compile(
    r"(?i)\b(how many|how much|score|accuracy|f1|precision|recall|bleu|rouge|"
    r"perplexity|percent|percentage|number|rate|size|higher|lower|highest|lowest|"
    r"improvement|result|results|performance)\b"
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?")
COMPARISON_RE = re.compile(
    r"(?i)\b(compare|comparison|different|difference|versus|vs\.?|better|worse|"
    r"outperform|baseline|than|between .+ and|trade[- ]?off)\b"
)
METHODOLOGY_RE = re.compile(
    r"(?i)\b(architecture|architectural|method|approach|model|algorithm|training|"
    r"objective|loss|encoder|decoder|embedding|attention|layer|pretrain|fine[- ]?tun|"
    r"implementation|implemented|parameter|pipeline|component)\b"
)
SYNTHESIS_RE = re.compile(
    r"(?i)\b(why|how|explain|summari[sz]e|summary|contribution|motivation|"
    r"limitation|advantage|disadvantage|main idea|overall|significance|challenge)\b"
)
YES_NO_RE = re.compile(r"(?i)^\s*(is|are|does|do|did|has|have|was|were|can|could|would|will)\b")


def iter_qasper_questions(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in load_qasper(path):
        for qa in paper.get("qas", []):
            answer_records = [
                a.get("answer") for a in qa.get("answers", [])
                if a.get("answer")
            ]
            evidence: list[str] = []
            reference_parts: list[str] = []
            answer_types: list[str] = []
            answer_form_labels: list[str] = []
            unanswerable_count = 0
            for ans in answer_records:
                if ans.get("unanswerable"):
                    unanswerable_count += 1
                    answer_form_labels.append("unanswerable")
                    continue
                evidence.extend(ans.get("evidence") or [])
                if ans.get("yes_no") is not None:
                    answer_types.append("yes_no")
                    yes_no_answer = "yes" if ans["yes_no"] else "no"
                    answer_form_labels.append(yes_no_answer)
                    reference_parts.append(yes_no_answer)
                elif ans.get("extractive_spans"):
                    answer_types.append("extractive")
                    answer_form_labels.append("extractive")
                    reference_parts.extend(ans.get("extractive_spans") or [])
                elif ans.get("free_form_answer"):
                    answer_types.append("free_form")
                    answer_form_labels.append("free_form")
                    reference_parts.append(ans["free_form_answer"])
                else:
                    answer_types.append("unknown")
                    answer_form_labels.append("unknown")

            rows.append({
                "qid": f"{paper['id']}__{qa.get('question_id')}",
                "paper_id": paper["id"],
                "question": qa.get("question", ""),
                "answer_types": sorted(set(answer_types)),
                "answer_form_labels": sorted(set(answer_form_labels)),
                "has_answer": bool(answer_types),
                "unanswerable_count": unanswerable_count,
                "reference_answer": " | ".join(reference_parts[:5]),
                "evidence": list(dict.fromkeys(e for e in evidence if isinstance(e, str))),
                "nlp_background": qa.get("nlp_background"),
                "topic_background": qa.get("topic_background"),
            })
    return rows


def proxy_label(row: dict[str, Any], mode: str = "question_only") -> str:
    question = row["question"]
    text_for_eval = question
    if mode == "answer_aware":
        text_for_eval = " ".join([
            question,
            row.get("reference_answer", ""),
            " ".join(row.get("evidence", [])[:3]),
        ])

    if NUMERIC_RE.search(question) or (
        mode == "answer_aware"
        and not re.search(r"(?i)\b(what|which)\s+(dataset|datasets|baseline|baselines|model|models|method|methods|language|languages)\b", question)
        and
        NUMBER_RE.search(text_for_eval)
        and re.search(r"(?i)\b(result|performance|score|accuracy|table|metric)\b", text_for_eval)
    ):
        return "numerical"

    if COMPARISON_RE.search(question):
        return "comparison"

    if METHODOLOGY_RE.search(question) and (
        re.search(r"(?i)\b(how|describe|what|which|explain)\b", question)
        or "method" in question.lower()
    ):
        return "methodology"

    if SYNTHESIS_RE.search(question):
        return "why_how"

    if YES_NO_RE.search(question):
        return "factual"

    return "factual"


def evaluate(rows: list[dict[str, Any]], proxy_mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        plan = route_query(row["question"], llm_client=None)
        expected = proxy_label(row, proxy_mode)
        out.append({
            "qid": row["qid"],
            "paper_id": row["paper_id"],
            "question": row["question"],
            "expected_proxy_query_type": expected,
            "predicted_query_type": plan.query_type,
            "correct": plan.query_type == expected,
            "route_level": plan.route_level,
            "classifier": plan.classifier,
            "route_confidence": plan.route_confidence,
            "retrieval_strategy": plan.retrieval_strategy,
            "evidence_need": plan.evidence_need,
            "answer_style": plan.answer_style,
            "answer_types": row["answer_types"],
            "answer_form_labels": row["answer_form_labels"],
            "has_answer": row["has_answer"],
            "unanswerable_count": row["unanswerable_count"],
            "proxy_mode": proxy_mode,
        })
    return out


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "qasper_router_debug.jsonl"
    summary_path = output_dir / "qasper_router_debug_summary.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(results)
    correct = sum(row["correct"] for row in results)
    confusion = defaultdict(Counter)
    pred_counts = Counter(row["predicted_query_type"] for row in results)
    expected_counts = Counter(row["expected_proxy_query_type"] for row in results)
    for row in results:
        confusion[row["expected_proxy_query_type"]][row["predicted_query_type"]] += 1

    lines = [
        "# QASPER Router Debug",
        "",
        "Important: this is proxy-label accuracy for retrieval intent.",
        "QASPER has official answer-form labels, but not gold labels for this project's router categories.",
        "Official answer-form labels are retained in the JSONL as `answer_form_labels`.",
        f"Proxy mode: `{results[0]['proxy_mode'] if results else 'unknown'}`.",
        "The router receives only the question. Default proxy labels are also question-only.",
        "",
        f"- Total questions: {total}",
        f"- Correct vs proxy: {correct}",
        f"- Proxy accuracy: {correct / total:.3f}",
        "",
        "## Expected Proxy Distribution",
        "",
    ]
    for label in QUERY_TYPES:
        lines.append(f"- {label}: {expected_counts[label]}")

    lines.extend(["", "## Predicted Distribution", ""])
    for label in QUERY_TYPES:
        lines.append(f"- {label}: {pred_counts[label]}")

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "| expected \\ predicted | " + " | ".join(QUERY_TYPES) + " |",
        "|---|" + "|".join("---" for _ in QUERY_TYPES) + "|",
    ])
    for expected in QUERY_TYPES:
        cells = [str(confusion[expected][predicted]) for predicted in QUERY_TYPES]
        lines.append(f"| {expected} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-Class Proxy Accuracy", ""])
    for expected in QUERY_TYPES:
        total_class = sum(confusion[expected].values())
        correct_class = confusion[expected][expected]
        acc = correct_class / total_class if total_class else 0
        lines.append(f"- {expected}: {correct_class}/{total_class} = {acc:.3f}")

    errors = [row for row in results if not row["correct"]]
    lines.extend(["", f"## First Errors ({len(errors)} total)", ""])
    for row in errors[:100]:
        lines.append(
            f"- `{row['qid']}` expected `{row['expected_proxy_query_type']}`, "
            f"got `{row['predicted_query_type']}`: {row['question']}"
        )
    if len(errors) > 100:
        lines.append(f"- ... {len(errors) - 100} more errors in `{jsonl_path.name}`")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(f"Proxy accuracy: {correct}/{total} = {correct / total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper", default="data/qasper_raw/qasper-dev-v0.3.json")
    parser.add_argument("--output-dir", default="data/runs/qasper_router_debug")
    parser.add_argument(
        "--proxy-mode",
        choices=["question_only", "answer_aware"],
        default="question_only",
        help="question_only avoids answer/evidence leakage; answer_aware is diagnostic only.",
    )
    args = parser.parse_args()
    rows = iter_qasper_questions(args.qasper)
    results = evaluate(rows, args.proxy_mode)
    write_outputs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
