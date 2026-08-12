import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from bench.dataset import load_qasper, filter_papers, extract_answerable_queries
from bench.chunking import build_chunks, build_qrels, filter_noise_chunks
from bench.index import BM25Index, DenseIndex
from bench.retrievers import (
    CERetriever,
    MultiQueryFusionCERetriever,
    RRFCEReciprocalRetriever,
    retrieve_bm25,
    retrieve_dense,
    retrieve_rrf,
)
from bench.query_classifier import classify_all
from bench.metrics import evaluate_strategy
from bench.llm_judge import stratified_sample, run_two_pass_judge
from bench.report import generate_report
from bench.config import TOP_K, LLM_PER_TYPE


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _make_run_paths(max_papers, no_llm, run_id=None, output_root="data/runs"):
    if run_id is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        size = "all" if max_papers is None else str(max_papers)
        judge = "no_llm" if no_llm else "llm"
        run_id = f"{stamp}_papers-{size}_{judge}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def run(max_papers=None, no_llm=False, llm_client=None, from_cache=None,
        run_id=None, output_root="data/runs", preserve_latest=True,
        strategies_filter=None):
    run_id, run_dir = _make_run_paths(max_papers, no_llm, run_id, output_root)
    if from_cache:
        print(f"Loading cached results from {from_cache}")
        with open(from_cache, encoding="utf-8") as f:
            cached = json.load(f)
        generate_report(
            [],
            [],
            [],
            output_json=str(run_dir / "results.json"),
            output_md=str(run_dir / "report.md"),
        )
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "from_cache": from_cache,
                "git_commit": _git_commit(),
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved run artifacts to: {run_dir}")
        return

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading QASPER validation split...")
    ds = load_qasper()
    papers = filter_papers(ds)
    if max_papers:
        papers = papers[:max_papers]
    print(f"Papers after filter: {len(papers)}")

    queries = extract_answerable_queries(papers)
    print(f"Answerable queries: {len(queries)}")

    chunks = build_chunks(papers)
    print(f"Chunks: {len(chunks)}")

    qrels = build_qrels(queries, chunks)
    chunks = filter_noise_chunks(chunks, qrels)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    query_type_map = classify_all(queries)
    for q in queries:
        q["query_type"] = query_type_map.get(q["qid"], "factual")

    # ── Indexes ───────────────────────────────────────────────────────────
    print("\nBuilding BM25 index...")
    bm25_index = BM25Index(chunks)

    print("Building Dense index (may take a few minutes)...")
    dense_index = DenseIndex(chunks)

    print("Loading CrossEncoder...")
    ce_retriever = CERetriever()
    rrf_ce_retriever = RRFCEReciprocalRetriever(ce_model=ce_retriever.ce_model)
    fusion_ce_retriever = MultiQueryFusionCERetriever(ce_model=ce_retriever.ce_model)

    # ── Strategy closures ─────────────────────────────────────────────────
    strategies = {
        "BM25":  lambda q: retrieve_bm25(q, bm25_index),
        "Dense": lambda q: retrieve_dense(q, dense_index),
        "RRF":   lambda q: retrieve_rrf(q, bm25_index, dense_index),
        "CE":    lambda q: ce_retriever.retrieve(q, bm25_index, dense_index, chunks_by_id),
        "RRF+CE": lambda q: rrf_ce_retriever.retrieve(q, bm25_index, dense_index, chunks_by_id),
        "Fusion+CE": lambda q: fusion_ce_retriever.retrieve(q, bm25_index, dense_index, chunks_by_id),
    }
    if strategies_filter:
        unknown = [name for name in strategies_filter if name not in strategies]
        if unknown:
            raise ValueError(
                "Unknown strategy name(s): "
                + ", ".join(unknown)
                + ". Valid names: "
                + ", ".join(strategies)
            )
        strategies = {name: strategies[name] for name in strategies_filter}

    # ── Evaluate Tier 1 ───────────────────────────────────────────────────
    all_metrics = []
    for name, fn in strategies.items():
        print(f"\nEvaluating {name}...")
        sm = evaluate_strategy(fn, queries, qrels, chunks_by_id, query_type_map, name=name)
        print(f"  Hit@5={sm.hit5:.4f}  EvidenceRecall@5={sm.evidence_recall5:.4f}  "
              f"MRR={sm.mrr:.4f}  NDCG@5={sm.ndcg5:.4f}  "
              f"Latency={sm.latency_ms:.1f}ms")
        all_metrics.append(sm)

    # ── LLM Judge Tier 2 & 3 (all strategies, shared sample) ─────────────
    sufficiency_maps = [{} for _ in all_metrics]
    badcase_pools   = [[] for _ in all_metrics]

    if not no_llm and llm_client is not None:
        # Sample once from the query set (use any strategy's results for metadata)
        sampled = stratified_sample(all_metrics[0].results, n_per_type=LLM_PER_TYPE)
        sampled_qids = [r.qid for r in sampled]
        print(f"\nRunning LLM judge on {len(sampled)} queries × {len(all_metrics)} strategies...")

        # Build qid -> question map for lookup
        qid_to_question = {r.qid: r.question for r in all_metrics[0].results}

        for idx, sm in enumerate(all_metrics):
            print(f"\n  Strategy: {sm.name}")
            # Build a dict qid -> QueryResult for this strategy
            result_by_qid = {r.qid: r for r in sm.results}
            # Get the sampled results for this strategy
            sampled_for_strategy = [
                result_by_qid[qid] for qid in sampled_qids
                if qid in result_by_qid
            ]

            suf_map, badcases = run_two_pass_judge(
                sm.results, sampled_for_strategy, chunks_by_id, llm_client
            )
            sufficiency_maps[idx] = suf_map
            badcase_pools[idx]    = badcases

    # ── Report ────────────────────────────────────────────────────────────
    output_json = run_dir / "results.json"
    output_md = run_dir / "report.md"
    generate_report(
        all_metrics,
        sufficiency_maps,
        badcase_pools,
        output_json=str(output_json),
        output_md=str(output_md),
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "args": {
            "max_papers": max_papers,
            "no_llm": no_llm,
            "llm_per_type": LLM_PER_TYPE,
            "top_k": TOP_K,
            "strategies_filter": strategies_filter,
        },
        "dataset": {
            "papers_after_filter": len(papers),
            "answerable_queries": len(queries),
            "qrels": len(qrels),
            "chunks_after_noise_filter": len(chunks),
        },
        "strategies": [sm.name for sm in all_metrics],
        "outputs": {
            "report": str(output_md),
            "results": str(output_json),
        },
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {run_dir / 'manifest.json'}")

    if preserve_latest:
        shutil.copyfile(output_json, "data/results_latest.json")
        shutil.copyfile(output_md, "data/report.md")
        shutil.copyfile(run_dir / "manifest.json", "data/latest_manifest.json")
        print("Updated latest shortcuts: data/results_latest.json, data/report.md")

    print(f"Run artifacts preserved in: {run_dir}")
    print("\nDone.")
