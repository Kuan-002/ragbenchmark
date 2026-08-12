"""Sweep local router-model thresholds on QASPER proxy labels.

This does not train the model. It evaluates a production-style policy:
rules route first, then the local semantic model may override only when the
rule result is factual and the model is confident about a non-factual label.

QASPER has official answer-form labels, commonly treated as five classes when
yes/no is split: extractive, free_form, yes, no, unanswerable. Those labels are
not the same as this project's retrieval-intent router categories, so this
script uses the same question-only proxy labels as scripts/debug_qasper_router.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperpilot.core.router import _rule_route_query
from paperpilot.core.router_model import LABELS, predict_query_type
from scripts.debug_qasper_router import iter_qasper_questions, proxy_label


def rows_with_predictions(qasper_path: str, proxy_mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in iter_qasper_questions(qasper_path):
        question = row["question"]
        base_plan = _rule_route_query(question, llm_client=None)
        semantic = predict_query_type(question)
        rows.append({
            "qid": row["qid"],
            "question": question,
            "expected": proxy_label(row, proxy_mode),
            "base": base_plan.query_type,
            "semantic": semantic,
        })
    return rows


def predict(row: dict[str, Any], score_threshold: float, margin_threshold: float) -> str:
    base = row["base"]
    semantic = row["semantic"]
    if not semantic:
        return base
    label = semantic["label"]
    if (
        base == "factual"
        and label != "factual"
        and float(semantic["score"]) >= score_threshold
        and float(semantic["margin"]) >= margin_threshold
    ):
        return label
    return base


def predict_model_only(row: dict[str, Any]) -> str:
    semantic = row["semantic"]
    if not semantic:
        return row["base"]
    return semantic["label"]


def summarize_predictions(rows: list[dict[str, Any]], name: str, predictions: list[str]) -> dict[str, Any]:
    total = len(rows)
    correct = 0
    confusion = defaultdict(Counter)
    for row, pred in zip(rows, predictions):
        correct += int(pred == row["expected"])
        confusion[row["expected"]][pred] += 1
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "confusion": {
            expected: {pred: confusion[expected][pred] for pred in LABELS}
            for expected in LABELS
        },
    }


def summarize(rows: list[dict[str, Any]], score: float, margin: float) -> dict[str, Any]:
    predictions = [predict(row, score, margin) for row in rows]
    out = summarize_predictions(rows, "rules_plus_model_override", predictions)
    out.update({
        "score_threshold": score,
        "margin_threshold": margin,
        "overrides": sum(int(pred != row["base"]) for row, pred in zip(rows, predictions)),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper", default="data/qasper_raw/qasper-dev-v0.3.json")
    parser.add_argument("--proxy-mode", choices=["question_only", "answer_aware"], default="question_only")
    parser.add_argument("--output", default="data/runs/qasper_router_threshold_sweep.json")
    parser.add_argument("--scores", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--margins", default="0.015,0.03,0.05,0.08,0.10,0.12,0.15")
    args = parser.parse_args()

    rows = rows_with_predictions(args.qasper, args.proxy_mode)
    score_values = [float(x) for x in args.scores.split(",") if x.strip()]
    margin_values = [float(x) for x in args.margins.split(",") if x.strip()]
    results = [
        summarize(rows, score, margin)
        for score in score_values
        for margin in margin_values
    ]
    results.sort(key=lambda item: (item["accuracy"], -item["overrides"]), reverse=True)
    baselines = [
        summarize_predictions(rows, "rules_only", [row["base"] for row in rows]),
        summarize_predictions(rows, "model_only", [predict_model_only(row) for row in rows]),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "proxy_mode": args.proxy_mode,
        "total_questions": len(rows),
        "baselines": baselines,
        "top_results": results[:20],
        "all_results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output}")
    for item in baselines:
        print(f"{item['name']}: acc={item['accuracy']:.3f} correct={item['correct']}/{item['total']}")
    for item in results[:10]:
        print(
            f"acc={item['accuracy']:.3f} correct={item['correct']}/{item['total']} "
            f"overrides={item['overrides']} score>={item['score_threshold']:.3f} "
            f"margin>={item['margin_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
