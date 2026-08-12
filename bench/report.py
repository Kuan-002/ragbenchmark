import json
from collections import Counter, defaultdict
from datetime import datetime

from bench.llm_judge import is_fixable


def _pct(n, total):
    return f"{n / total * 100:.0f}% ({n}/{total})" if total else "N/A"


def _rank(value):
    return f"{value:.1f}" if value is not None else "N/A"


def _field_breakdown(results, group_key):
    groups = defaultdict(list)
    for r in results:
        val = getattr(r, group_key, None) or "unknown"
        groups[val].append(r)
    return {
        k: {
            "n": len(v),
            "hit5": sum(r.hit5 for r in v) / len(v),
            "mrr": sum(r.mrr for r in v) / len(v),
            "evidence_recall5": sum(r.evidence_recall5 for r in v) / len(v),
            "ndcg5": sum(r.ndcg5 for r in v) / len(v),
        }
        for k, v in sorted(groups.items())
    }


def generate_report(strategy_metrics_list, sufficiency_maps, badcase_pools,
                    output_json="data/results_latest.json",
                    output_md="data/report.md"):
    dump = {}
    for sm in strategy_metrics_list:
        dump[sm.name] = {
            "hit1": sm.hit1,
            "hit3": sm.hit3,
            "hit5": sm.hit5,
            "mrr": sm.mrr,
            "mean_gold_rank": sm.mean_gold_rank,
            "median_gold_rank": sm.median_gold_rank,
            "evidence_recall5": sm.evidence_recall5,
            "all_evidence_recall5": sm.all_evidence_recall5,
            "ndcg5": sm.ndcg5,
            "latency_ms": sm.latency_ms,
            "type_breakdown": sm.type_breakdown(),
            "evidence_count_breakdown": sm.evidence_count_breakdown(),
            "extra_stats": sm.extra_stats,
            "results": [r.to_dict() for r in sm.results],
        }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {output_json}")

    lines = [
        "# RAG Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Metric Framing",
        "",
        "QASPER is used here as an evidence-grounded RAG evaluation set. "
        "Most aligned questions have one or a small number of gold evidence chunks, "
        "so Hit@k, MRR, Gold Rank, and Evidence Recall@k are the primary retrieval "
        "diagnostics. NDCG@5 is retained as a secondary ranking metric.",
        "",
        "## Tier 1 - Evidence Retrieval",
        "",
        "| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Evidence Recall@5 | All Evidence@5 | Median Gold Rank | NDCG@5 | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sm in strategy_metrics_list:
        lines.append(
            f"| {sm.name} | {sm.hit1:.4f} | {sm.hit3:.4f} | {sm.hit5:.4f} "
            f"| {sm.mrr:.4f} | {sm.evidence_recall5:.4f} "
            f"| {sm.all_evidence_recall5:.4f} | {_rank(sm.median_gold_rank)} "
            f"| {sm.ndcg5:.4f} | {sm.latency_ms:.1f} |"
        )

    all_types = sorted({qt for sm in strategy_metrics_list for qt in sm.type_breakdown()})
    lines += ["", "## Tier 1.5 - Hit@5 by Query Type", ""]
    header = "| Type | N |" + "".join(f" {sm.name} Hit@5 |" for sm in strategy_metrics_list)
    sep = "|---|---|" + "".join("---|" for _ in strategy_metrics_list)
    lines += [header, sep]
    for qt in all_types:
        n = sum(1 for r in strategy_metrics_list[0].results if r.query_type == qt)
        row = f"| {qt} | {n} |"
        for sm in strategy_metrics_list:
            row += f" {sm.type_breakdown().get(qt, {}).get('hit5', 0.0):.4f} |"
        lines.append(row)

    lines += ["", "## Tier 1.6 - Single vs Multi Evidence", ""]
    lines += [
        "| Strategy | Group | N | Hit@5 | MRR | Evidence Recall@5 | All Evidence@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sm in strategy_metrics_list:
        for group, row in sm.evidence_count_breakdown().items():
            lines.append(
                f"| {sm.name} | {group} | {row['n']} | {row['hit5']:.4f} "
                f"| {row['mrr']:.4f} | {row['evidence_recall5']:.4f} "
                f"| {row['all_evidence_recall5']:.4f} |"
            )

    ce_sm = next((sm for sm in strategy_metrics_list if sm.name == "CE"), None)
    if ce_sm:
        lines += ["", "## Tier 1.7 - CE Stratified Diagnostics", ""]
        for field, label in [
            ("answer_type", "Answer Type"),
            ("nlp_background", "NLP Background"),
            ("topic_background", "Topic Background"),
        ]:
            lines += [f"### {label}", ""]
            lines += [
                "| Value | N | Hit@5 | MRR | Evidence Recall@5 | NDCG@5 |",
                "|---|---|---|---|---|---|",
            ]
            for val, row in sorted(_field_breakdown(ce_sm.results, field).items(),
                                   key=lambda x: -x[1]["hit5"]):
                lines.append(
                    f"| {val} | {row['n']} | {row['hit5']:.4f} | {row['mrr']:.4f} "
                    f"| {row['evidence_recall5']:.4f} | {row['ndcg5']:.4f} |"
                )
            lines.append("")

    if any(sufficiency_maps):
        lines += ["## Tier 2 - Context Sufficiency", ""]
        lines += [
            "| Strategy | Avg Score | Fully Sufficient | Partially Sufficient | Insufficient |",
            "|---|---|---|---|---|",
        ]
        for sm, suf_map in zip(strategy_metrics_list, sufficiency_maps):
            if not suf_map:
                lines.append(f"| {sm.name} | N/A | N/A | N/A | N/A |")
                continue
            n = len(suf_map)
            details = [
                v if isinstance(v, dict) else {"verdict": v, "score": None}
                for v in suf_map.values()
            ]
            scores = [d["score"] for d in details if isinstance(d.get("score"), int)]
            avg_score = f"{sum(scores) / len(scores):.2f}" if scores else "N/A"
            fully = sum(1 for d in details if d.get("verdict") == "FULLY_SUFFICIENT")
            partial = sum(1 for d in details if d.get("verdict") == "PARTIALLY_SUFFICIENT")
            insuf = sum(1 for d in details if d.get("verdict") == "INSUFFICIENT")
            lines.append(
                f"| {sm.name} | {avg_score} | {_pct(fully, n)} | "
                f"{_pct(partial, n)} | {_pct(insuf, n)} |"
            )

    if any(badcase_pools):
        lines += ["", "## Tier 3 - Badcase Analysis", ""]
        lines += [
            "| Strategy | N Badcases | Fixable | RETRIEVAL_FAILURE | MISSING_EVIDENCE | REDUNDANCY |",
            "|---|---|---|---|---|---|",
        ]
        for sm, pool in zip(strategy_metrics_list, badcase_pools):
            if not pool:
                lines.append(f"| {sm.name} | 0 | N/A | N/A | N/A | N/A |")
                continue
            cats = Counter(r.badcase_category for r in pool)
            fixable_n = sum(1 for r in pool if is_fixable(r.badcase_category))
            lines.append(
                f"| {sm.name} | {len(pool)} | {_pct(fixable_n, len(pool))} "
                f"| {cats.get('RETRIEVAL_FAILURE', 0)} "
                f"| {cats.get('MISSING_EVIDENCE', 0)} "
                f"| {cats.get('REDUNDANCY', 0)} |"
            )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report: {output_md}")
