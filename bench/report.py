import json
from collections import defaultdict
from datetime import datetime
from bench.llm_judge import is_fixable


def _pct(n, total):
    return f"{n/total*100:.0f}% ({n}/{total})" if total else "N/A"


def _tier16_breakdown(results, group_key):
    """NDCG@5 grouped by a metadata field."""
    groups = defaultdict(list)
    for r in results:
        val = getattr(r, group_key, None) or "unknown"
        groups[val].append(r.ndcg5)
    return {k: (sum(v)/len(v), len(v)) for k, v in sorted(groups.items())}


def generate_report(strategy_metrics_list, sufficiency_maps, badcase_pools,
                    output_json="data/results_latest.json",
                    output_md="data/report.md"):

    # ── JSON dump ────────────────────────────────────────────────────────
    dump = {}
    for sm in strategy_metrics_list:
        dump[sm.name] = {
            "ndcg5": sm.ndcg5,
            "recall5": sm.recall5,
            "mrr": sm.mrr,
            "latency_ms": sm.latency_ms,
            "type_breakdown": sm.type_breakdown(),
            "results": [r.to_dict() for r in sm.results],
        }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {output_json}")

    # ── Markdown ─────────────────────────────────────────────────────────
    lines = [
        "# RAG Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Tier 1 — Retrieval Metrics",
        "",
        "| Strategy | NDCG@5 | Recall@5 | MRR | Latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for sm in strategy_metrics_list:
        lines.append(
            f"| {sm.name} | {sm.ndcg5:.4f} | {sm.recall5:.4f} "
            f"| {sm.mrr:.4f} | {sm.latency_ms:.1f} |"
        )

    # ── Tier 1.5: by query type ───────────────────────────────────────────
    all_types = sorted({qt for sm in strategy_metrics_list for qt in sm.type_breakdown()})
    lines += ["", "## Tier 1.5 — NDCG@5 by Query Type", ""]
    header = "| Type | N |" + "".join(f" {sm.name} |" for sm in strategy_metrics_list)
    sep    = "|---|---|" + "".join("---|" for _ in strategy_metrics_list)
    lines += [header, sep]
    for qt in all_types:
        n   = sum(1 for r in strategy_metrics_list[0].results if r.query_type == qt)
        row = f"| {qt} | {n} |"
        for sm in strategy_metrics_list:
            row += f" {sm.type_breakdown().get(qt, 0.0):.4f} |"
        lines.append(row)

    # ── Tier 1.6: by answer_type / nlp_background / topic_background ─────
    lines += ["", "## Tier 1.6 — NDCG@5 Stratified Analysis (CE only)", ""]
    ce_sm = next((sm for sm in strategy_metrics_list if sm.name == "CE"), None)
    if ce_sm:
        for field, label in [
            ("answer_type",      "Answer Type"),
            ("nlp_background",   "NLP Background"),
            ("topic_background", "Topic Background"),
        ]:
            bd = _tier16_breakdown(ce_sm.results, field)
            lines += [f"### {label}", ""]
            lines += ["| Value | N | NDCG@5 |", "|---|---|---|"]
            for val, (ndcg, n) in sorted(bd.items(), key=lambda x: -x[1][0]):
                lines.append(f"| {val} | {n} | {ndcg:.4f} |")
            lines.append("")

    # ── Tier 2: LLM Sufficiency (all strategies) ─────────────────────────
    if any(sufficiency_maps):
        lines += ["## Tier 2 — LLM Sufficiency", ""]
        lines += ["| Strategy | Fully Sufficient | Partially Sufficient | Insufficient |",
                  "|---|---|---|---|"]
        for sm, suf_map in zip(strategy_metrics_list, sufficiency_maps):
            if not suf_map:
                lines.append(f"| {sm.name} | — | — | — |")
                continue
            n      = len(suf_map)
            fully  = sum(1 for v in suf_map.values() if v == "FULLY_SUFFICIENT")
            partial= sum(1 for v in suf_map.values() if v == "PARTIALLY_SUFFICIENT")
            insuf  = sum(1 for v in suf_map.values() if v == "INSUFFICIENT")
            lines.append(
                f"| {sm.name} | {_pct(fully, n)} | {_pct(partial, n)} | {_pct(insuf, n)} |"
            )

    # ── Tier 3: Badcase root cause (all strategies) ───────────────────────
    if any(badcase_pools):
        lines += ["", "## Tier 3 — Badcase Analysis", ""]
        lines += ["| Strategy | N Badcases | Fixable | RETRIEVAL_FAILURE | MISSING_EVIDENCE | REDUNDANCY |",
                  "|---|---|---|---|---|---|"]
        for sm, pool in zip(strategy_metrics_list, badcase_pools):
            if not pool:
                lines.append(f"| {sm.name} | 0 | — | — | — | — |")
                continue
            from collections import Counter
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
