"""Debug the paper-question router on 500 labeled synthetic questions.

This is an offline diagnostic: it calls route_query(..., llm_client=None) so
results are deterministic and do not depend on API keys. The goal is not to
prove benchmark quality, but to expose obvious routing failures before they
reach retrieval and answer generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperpilot.core.agentic_rag import _question_profile
from paperpilot.core.router import route_query


QUERY_TYPES = ["numerical", "comparison", "why_how", "methodology", "factual"]


def _cases_for(label: str, questions: Iterable[str]) -> list[dict]:
    return [
        {
            "id": f"{label}_{idx:03d}",
            "question": question,
            "expected_query_type": label,
        }
        for idx, question in enumerate(questions, 1)
    ]


def build_cases() -> list[dict]:
    numerical_subjects = [
        "BLEU score", "ROUGE-L score", "F1 score", "accuracy", "precision",
        "recall", "perplexity", "training time", "number of parameters",
        "dataset size", "batch size", "learning rate", "dropout rate",
        "latency", "memory usage", "throughput", "error rate", "win rate",
        "improvement percentage", "ablation score",
    ]
    numerical_templates = [
        "What {subject} does the proposed model report?",
        "What are the key experimental results and reported {subject}?",
        "How many {subject} are used in the experiment?",
        "Which setting has the highest {subject}?",
        "What is the difference in {subject} between the baseline and the proposed method?",
    ]

    comparison_subjects = [
        "Transformer and RNN", "BERT and ELMo", "the base and large model",
        "encoder-only and decoder-only architectures", "learned and sinusoidal position embeddings",
        "beam search and greedy decoding", "the proposed method and the baseline",
        "Table 1 and Table 2", "training from scratch and pretraining",
        "self-attention and convolution",
    ]
    comparison_templates = [
        "Compare {subject}.",
        "What is the difference between {subject}?",
        "Which performs better, {subject}?",
        "How does {subject} differ in the experiments?",
        "What advantages does one side have over the other for {subject}?",
        "Is {subject} better on the reported benchmark?",
        "Compare the strengths and weaknesses of {subject}.",
        "What evidence supports the comparison between {subject}?",
        "How does the paper evaluate {subject} against each other?",
        "What trade-off is reported between {subject}?",
    ]

    synthesis_subjects = [
        "core contribution", "main idea", "motivation", "limitation",
        "main finding", "overall approach", "central claim", "research problem",
        "paper's significance", "failure mode", "advantage", "design rationale",
        "error analysis", "discussion section", "conclusion", "future work",
        "why the method works", "how the evidence supports the claim",
        "what the paper is about", "what problem the paper solves",
    ]
    synthesis_templates = [
        "Summarize the {subject} of this paper.",
        "Explain the {subject} in plain terms.",
        "Why is the {subject} important?",
        "How does the paper justify its {subject}?",
        "What is the {subject}, and what evidence supports it?",
    ]

    methodology_subjects = [
        "model architecture", "encoder", "decoder", "attention layer",
        "training objective", "loss function", "pretraining task",
        "fine-tuning procedure", "embedding layer", "positional encoding",
        "optimization setup", "data preprocessing pipeline", "inference algorithm",
        "beam search procedure", "masking strategy", "feed-forward network",
        "residual connection", "normalization layer", "tokenization method",
        "implementation details",
    ]
    methodology_templates = [
        "Describe the {subject} step by step.",
        "How is the {subject} implemented?",
        "What role does the {subject} play in the method?",
        "Which components are used in the {subject}?",
        "Explain the {subject} used by the proposed model.",
    ]

    factual_subjects = [
        "paper title", "authors", "publication venue", "dataset name",
        "task name", "baseline name", "model name", "section title",
        "evaluation benchmark", "training corpus", "language pair",
        "source dataset", "target task", "appendix name", "figure caption",
        "table caption", "software library", "license statement",
        "conference year", "introduced acronym",
    ]
    factual_templates = [
        "What is the {subject}?",
        "Which {subject} is mentioned in the paper?",
        "Name the {subject} used in the experiment.",
        "Where does the paper state the {subject}?",
        "What does the paper call the {subject}?",
    ]

    cases: list[dict] = []
    cases.extend(_cases_for("numerical", _expand(numerical_templates, numerical_subjects)))
    cases.extend(_cases_for("comparison", _expand(comparison_templates, comparison_subjects)))
    cases.extend(_cases_for("why_how", _expand(synthesis_templates, synthesis_subjects)))
    cases.extend(_cases_for("methodology", _expand(methodology_templates, methodology_subjects)))
    cases.extend(_cases_for("factual", _expand(factual_templates, factual_subjects)))
    if len(cases) != 500:
        raise AssertionError(f"Expected 500 cases, got {len(cases)}")
    return cases


def _expand(templates: list[str], subjects: list[str]) -> list[str]:
    return [
        template.format(subject=subject)
        for subject in subjects
        for template in templates
    ]


def evaluate(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        plan = route_query(case["question"], llm_client=None)
        plan_dict = plan.to_dict()
        profile = _question_profile(case["question"], plan_dict)
        rows.append({
            **case,
            "predicted_query_type": plan.query_type,
            "correct": plan.query_type == case["expected_query_type"],
            "route_level": plan.route_level,
            "classifier": plan.classifier,
            "route_confidence": plan.route_confidence,
            "retrieval_strategy": plan.retrieval_strategy,
            "evidence_need": plan.evidence_need,
            "answer_style": plan.answer_style,
            "source_preference": plan.source_preference,
            "agent_suggested_start": profile["suggested_start"],
            "agent_numeric_or_table": profile["numeric_or_table"],
            "agent_synthesis_or_contribution": profile["synthesis_or_contribution"],
        })
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "router_debug_500.jsonl"
    summary_path = output_dir / "router_debug_500_summary.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    by_expected = defaultdict(Counter)
    for row in rows:
        by_expected[row["expected_query_type"]][row["predicted_query_type"]] += 1

    lines = [
        "# Router Debug 500",
        "",
        f"- Total: {total}",
        f"- Correct: {correct}",
        f"- Accuracy: {correct / total:.3f}",
        "",
        "## Confusion Matrix",
        "",
        "| expected \\ predicted | " + " | ".join(QUERY_TYPES) + " |",
        "|---|" + "|".join("---" for _ in QUERY_TYPES) + "|",
    ]
    for expected in QUERY_TYPES:
        cells = [str(by_expected[expected][predicted]) for predicted in QUERY_TYPES]
        lines.append(f"| {expected} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-Class Accuracy", ""])
    for expected in QUERY_TYPES:
        total_class = sum(by_expected[expected].values())
        correct_class = by_expected[expected][expected]
        lines.append(f"- {expected}: {correct_class}/{total_class} = {correct_class / total_class:.3f}")

    errors = [row for row in rows if not row["correct"]]
    lines.extend(["", f"## Errors ({len(errors)})", ""])
    for row in errors[:80]:
        lines.append(
            f"- `{row['id']}` expected `{row['expected_query_type']}`, "
            f"got `{row['predicted_query_type']}`: {row['question']}"
        )
    if len(errors) > 80:
        lines.append(f"- ... {len(errors) - 80} more errors in `{jsonl_path.name}`")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/runs/router_debug_500",
        help="Directory for JSONL details and Markdown summary.",
    )
    args = parser.parse_args()
    rows = evaluate(build_cases())
    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
