"""Run standalone agentic QA evaluation with full visible traces.

Example:
  python run_agentic_qa.py --max-papers 5 --limit 20 \
    --output data/runs/agentic_qa_smoke/trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from bench.agentic_eval import evaluate_agentic_qa
from bench.chunking import build_chunks, build_qrels, filter_noise_chunks
from bench.dataset import extract_answerable_queries, filter_papers, load_qasper
from paperpilot.core.agentic_rag import run_agentic_rag
from paperpilot.core.indexer import PaperIndex, load_models
from paperpilot.core.router import route_query


def _get_llm_client():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or DEEPSEEK_API_KEY before running agentic QA.")

    import openai

    base_url = os.environ.get("OPENAI_BASE_URL") or (
        "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
    )
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _select_queries(queries: list[dict], qrels: dict[str, set[str]],
                    limit: int | None) -> list[dict]:
    selected = [query for query in queries if query["qid"] in qrels]
    return selected[:limit] if limit else selected


def _write_metrics(metrics, output_path: Path, args: argparse.Namespace,
                   dataset_stats: dict) -> Path:
    metrics_path = output_path.with_suffix(".metrics.json")
    payload = metrics.to_dict()
    payload["run"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "max_papers": args.max_papers,
        "limit": args.limit,
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "trace_jsonl": str(output_path),
        "dataset": dataset_stats,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=5,
                        help="Number of filtered QASPER papers to evaluate.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum number of aligned QA examples to run.")
    parser.add_argument("--output", type=str, required=True,
                        help="JSONL path for per-question trace records.")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional model override. Defaults to OPENAI_MODEL or gpt-4o-mini.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Final evidence count passed to answer generation.")
    parser.add_argument("--max-steps", type=int, default=7,
                        help="Maximum tool-calling loop steps per question.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_llm_client()
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"Agentic QA model: {model}")

    print("Loading QASPER validation split...")
    papers = filter_papers(load_qasper())
    if args.max_papers:
        papers = papers[:args.max_papers]
    print(f"Papers: {len(papers)}")

    queries = extract_answerable_queries(papers)
    chunks = build_chunks(papers)
    qrels = build_qrels(queries, chunks)
    chunks = filter_noise_chunks(chunks, qrels)
    chunks_by_paper: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_paper.setdefault(chunk["paper_id"], []).append(chunk)

    selected = _select_queries(queries, qrels, args.limit)
    print(f"Aligned QA examples: {len(selected)}")

    print("Loading retrieval models...")
    load_models()
    indexes: dict[str, PaperIndex] = {}
    chunk_positions: dict[str, tuple[str, int]] = {}
    for paper_id, paper_chunks in chunks_by_paper.items():
        for idx, chunk in enumerate(paper_chunks):
            chunk_positions[chunk["chunk_id"]] = (paper_id, idx)

    def paper_index(paper_id: str) -> PaperIndex:
        if paper_id not in indexes:
            indexes[paper_id] = PaperIndex(chunks_by_paper.get(paper_id, []))
        return indexes[paper_id]

    def agent_runner(query: dict) -> dict:
        question = query["question"]
        query_plan = route_query(question, llm_client=None).to_dict()
        index = paper_index(query["paper_id"])

        def retrieve_fn(search_query: str, top_k: int):
            strategy = query_plan.get("retrieval_strategy")
            if strategy in {"fusion", "hyde", "decompose"}:
                return index.retrieve_multi_query_ce(search_query, query_plan, top_k=top_k)
            return index.retrieve_ce(
                search_query,
                top_k=top_k,
                cross_table=query_plan.get("cross_table", False),
                query_type=query_plan.get("query_type", "factual"),
            )

        def neighbor_fn(chunk_id: str, window: int):
            if chunk_id not in chunk_positions:
                return []
            paper_id, idx = chunk_positions[chunk_id]
            paper_chunks = chunks_by_paper.get(paper_id, [])
            start = max(0, idx - window)
            end = min(len(paper_chunks), idx + window + 1)
            anchor_score = 0.0
            rows = []
            for pos in range(start, end):
                chunk = paper_chunks[pos]
                if chunk["chunk_id"] == chunk_id:
                    continue
                distance = abs(pos - idx)
                rows.append({
                    "chunk": chunk,
                    "score": anchor_score - 0.01 * distance,
                })
            return rows

        return run_agentic_rag(
            question,
            query_plan,
            retrieve_fn,
            client,
            top_k=args.top_k,
            neighbor_fn=neighbor_fn,
            max_steps=args.max_steps,
            model=model,
        )

    metrics = evaluate_agentic_qa(
        agent_runner,
        selected,
        qrels,
        name="Agentic-RAG",
        k=args.top_k,
        trace_jsonl=output_path,
    )
    metrics_path = _write_metrics(metrics, output_path, args, {
        "papers": len(papers),
        "queries": len(queries),
        "aligned_queries": len(selected),
        "chunks": len(chunks),
        "qrels": len(qrels),
    })

    print(
        "Done: "
        f"N={metrics.n} Hit@1={metrics.hit1:.4f} Hit@5={metrics.hit5:.4f} "
        f"EvidenceRecall@5={metrics.evidence_recall5:.4f} "
        f"MRR={metrics.mrr:.4f} Latency={metrics.latency_ms:.1f}ms"
    )
    print(f"Trace JSONL: {output_path}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
