import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional


def ndcg_at_k(retrieved_ids, gold_ids, k=5):
    gold_set = set(gold_ids)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, cid in enumerate(retrieved_ids[:k])
        if cid in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids, gold_ids, k=5):
    gold_set = set(gold_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in gold_set)
    return hits / len(gold_set) if gold_set else 0.0


def mrr(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids):
        if cid in gold_set:
            return 1.0 / (rank + 1)
    return 0.0


def section_redundancy(retrieved_ids, chunks_by_id, k=5):
    sections = [
        chunks_by_id[cid]["section"]
        for cid in retrieved_ids[:k]
        if cid in chunks_by_id
    ]
    if not sections:
        return 0.0
    return 1.0 - len(set(sections)) / len(sections)


@dataclass
class QueryResult:
    qid: str
    question: str
    query_type: str
    answer_type: Optional[str]
    nlp_background: Optional[str]
    topic_background: Optional[str]
    ndcg5: float
    recall5: float
    mrr: float
    sec_redundancy: float
    retrieved_ids: List[str]
    gold_ids: List[str]
    latency_ms: float
    sufficiency: Optional[str] = None
    badcase_category: Optional[str] = None

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class StrategyMetrics:
    name: str
    ndcg5: float
    recall5: float
    mrr: float
    latency_ms: float
    results: List[QueryResult] = field(default_factory=list)

    def type_breakdown(self):
        from collections import defaultdict
        by_type = defaultdict(list)
        for r in self.results:
            by_type[r.query_type].append(r.ndcg5)
        return {t: sum(v) / len(v) for t, v in by_type.items()}


def evaluate_strategy(strategy_fn, queries, qrels, chunks_by_id,
                      query_type_map, k=5, name=""):
    results = []
    latencies = []

    for query in queries:
        qid = query["qid"]
        if qid not in qrels:
            continue

        t0 = time.time()
        retrieved = strategy_fn(query["question"])
        latency = (time.time() - t0) * 1000

        retrieved_ids = [cid for cid, _ in retrieved]
        gold_ids = list(qrels[qid])

        results.append(QueryResult(
            qid=qid,
            question=query["question"],
            query_type=query_type_map.get(qid, "factual"),
            answer_type=query.get("answer_type"),
            nlp_background=query.get("nlp_background"),
            topic_background=query.get("topic_background"),
            ndcg5=ndcg_at_k(retrieved_ids, gold_ids, k),
            recall5=recall_at_k(retrieved_ids, gold_ids, k),
            mrr=mrr(retrieved_ids, gold_ids),
            sec_redundancy=section_redundancy(retrieved_ids, chunks_by_id, k),
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            latency_ms=latency,
        ))
        latencies.append(latency)

    n = len(results)
    return StrategyMetrics(
        name=name,
        ndcg5=sum(r.ndcg5 for r in results) / n,
        recall5=sum(r.recall5 for r in results) / n,
        mrr=sum(r.mrr for r in results) / n,
        latency_ms=sum(latencies) / len(latencies),
        results=results,
    )
