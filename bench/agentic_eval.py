"""Standalone QA ground-truth evaluation for agentic RAG runs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bench.metrics import (
    all_evidence_recall_at_k,
    first_gold_rank,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


AgentRunner = Callable[[dict], dict]


@dataclass
class AgenticQaResult:
    qid: str
    question: str
    reference_answer: str | None
    retrieved_ids: list[str]
    gold_ids: list[str]
    answer: str
    hit1: float
    hit3: float
    hit5: float
    evidence_recall5: float
    all_evidence_recall5: float
    mrr: float
    ndcg5: float
    first_gold_rank: int | None
    latency_ms: float
    trace: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AgenticQaMetrics:
    name: str
    n: int
    hit1: float
    hit3: float
    hit5: float
    evidence_recall5: float
    all_evidence_recall5: float
    mrr: float
    ndcg5: float
    latency_ms: float
    results: list[AgenticQaResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "hit1": self.hit1,
            "hit3": self.hit3,
            "hit5": self.hit5,
            "evidence_recall5": self.evidence_recall5,
            "all_evidence_recall5": self.all_evidence_recall5,
            "mrr": self.mrr,
            "ndcg5": self.ndcg5,
            "latency_ms": self.latency_ms,
            "results": [row.to_dict() for row in self.results],
        }


def _retrieved_ids(agent_result: dict) -> list[str]:
    ids: list[str] = []
    for row in agent_result.get("retrieved", []):
        cid = row.get("chunk", {}).get("chunk_id")
        if cid:
            ids.append(cid)
    return ids


def evaluate_agentic_qa(agent_runner: AgentRunner, queries: list[dict],
                        qrels: dict[str, set[str]], *,
                        name: str = "Agentic-RAG", k: int = 5,
                        trace_jsonl: str | Path | None = None) -> AgenticQaMetrics:
    """Evaluate agentic QA runs against gold evidence qrels.

    `agent_runner` receives the full query dict and returns the result from
    `paperpilot.core.agentic_rag.run_agentic_rag`. Every per-query record keeps
    the complete visible agent rounds and trace for later auditing.
    """
    output_path = Path(trace_jsonl) if trace_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    results: list[AgenticQaResult] = []
    latencies: list[float] = []
    for query in queries:
        qid = query["qid"]
        if qid not in qrels:
            continue

        started = time.time()
        agent_result = agent_runner(query)
        latency_ms = (time.time() - started) * 1000
        latencies.append(latency_ms)

        retrieved_ids = _retrieved_ids(agent_result)
        gold_ids = list(qrels[qid])
        rank = first_gold_rank(retrieved_ids, gold_ids)

        row = AgenticQaResult(
            qid=qid,
            question=query["question"],
            reference_answer=query.get("reference_answer"),
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            answer=agent_result.get("answer", ""),
            hit1=hit_at_k(retrieved_ids, gold_ids, 1),
            hit3=hit_at_k(retrieved_ids, gold_ids, 3),
            hit5=hit_at_k(retrieved_ids, gold_ids, k),
            evidence_recall5=recall_at_k(retrieved_ids, gold_ids, k),
            all_evidence_recall5=all_evidence_recall_at_k(retrieved_ids, gold_ids, k),
            mrr=mrr(retrieved_ids, gold_ids),
            ndcg5=ndcg_at_k(retrieved_ids, gold_ids, k),
            first_gold_rank=rank,
            latency_ms=latency_ms,
            trace=agent_result.get("trace", []),
            rounds=agent_result.get("rounds", []),
        )
        results.append(row)

        if output_path:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    n = len(results)
    if n == 0:
        return AgenticQaMetrics(name=name, n=0, hit1=0, hit3=0, hit5=0,
                                evidence_recall5=0, all_evidence_recall5=0,
                                mrr=0, ndcg5=0, latency_ms=0, results=[])

    return AgenticQaMetrics(
        name=name,
        n=n,
        hit1=sum(row.hit1 for row in results) / n,
        hit3=sum(row.hit3 for row in results) / n,
        hit5=sum(row.hit5 for row in results) / n,
        evidence_recall5=sum(row.evidence_recall5 for row in results) / n,
        all_evidence_recall5=sum(row.all_evidence_recall5 for row in results) / n,
        mrr=sum(row.mrr for row in results) / n,
        ndcg5=sum(row.ndcg5 for row in results) / n,
        latency_ms=sum(latencies) / len(latencies),
        results=results,
    )
