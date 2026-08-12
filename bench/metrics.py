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


def hit_at_k(retrieved_ids, gold_ids, k=5):
    gold_set = set(gold_ids)
    return 1.0 if any(cid in gold_set for cid in retrieved_ids[:k]) else 0.0


def recall_at_k(retrieved_ids, gold_ids, k=5):
    gold_set = set(gold_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in gold_set)
    return hits / len(gold_set) if gold_set else 0.0


def all_evidence_recall_at_k(retrieved_ids, gold_ids, k=5):
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    return 1.0 if gold_set.issubset(set(retrieved_ids[:k])) else 0.0


def mrr(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids):
        if cid in gold_set:
            return 1.0 / (rank + 1)
    return 0.0


def first_gold_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_set:
            return rank
    return None


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
    reference_answer: Optional[str]
    query_type: str
    answer_type: Optional[str]
    nlp_background: Optional[str]
    topic_background: Optional[str]
    hit1: float
    hit3: float
    hit5: float
    ndcg5: float
    evidence_recall5: float
    all_evidence_recall5: float
    mrr: float
    first_gold_rank: Optional[int]
    gold_count: int
    sec_redundancy: float
    retrieved_ids: List[str]
    gold_ids: List[str]
    latency_ms: float
    sufficiency: Optional[str] = None
    sufficiency_score: Optional[int] = None
    sufficiency_reason: Optional[str] = None
    sufficiency_missing: List[str] = field(default_factory=list)
    badcase_category: Optional[str] = None

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class StrategyMetrics:
    name: str
    hit1: float
    hit3: float
    hit5: float
    ndcg5: float
    evidence_recall5: float
    all_evidence_recall5: float
    mrr: float
    mean_gold_rank: Optional[float]
    median_gold_rank: Optional[float]
    latency_ms: float
    results: List[QueryResult] = field(default_factory=list)
    extra_stats: Dict = field(default_factory=dict)

    def type_breakdown(self):
        from collections import defaultdict
        by_type = defaultdict(list)
        for r in self.results:
            by_type[r.query_type].append(r)
        return {
            t: {
                "n": len(v),
                "hit5": sum(r.hit5 for r in v) / len(v),
                "mrr": sum(r.mrr for r in v) / len(v),
                "evidence_recall5": sum(r.evidence_recall5 for r in v) / len(v),
                "ndcg5": sum(r.ndcg5 for r in v) / len(v),
            }
            for t, v in by_type.items()
        }

    def evidence_count_breakdown(self):
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            label = "single" if r.gold_count == 1 else "multi"
            groups[label].append(r)
        return {
            label: {
                "n": len(rows),
                "hit5": sum(r.hit5 for r in rows) / len(rows),
                "mrr": sum(r.mrr for r in rows) / len(rows),
                "evidence_recall5": sum(r.evidence_recall5 for r in rows) / len(rows),
                "all_evidence_recall5": sum(r.all_evidence_recall5 for r in rows) / len(rows),
            }
            for label, rows in sorted(groups.items())
        }


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
        rank = first_gold_rank(retrieved_ids, gold_ids)

        results.append(QueryResult(
            qid=qid,
            question=query["question"],
            reference_answer=query.get("reference_answer"),
            query_type=query_type_map.get(qid, "factual"),
            answer_type=query.get("answer_type"),
            nlp_background=query.get("nlp_background"),
            topic_background=query.get("topic_background"),
            hit1=hit_at_k(retrieved_ids, gold_ids, 1),
            hit3=hit_at_k(retrieved_ids, gold_ids, 3),
            hit5=hit_at_k(retrieved_ids, gold_ids, k),
            ndcg5=ndcg_at_k(retrieved_ids, gold_ids, k),
            evidence_recall5=recall_at_k(retrieved_ids, gold_ids, k),
            all_evidence_recall5=all_evidence_recall_at_k(retrieved_ids, gold_ids, k),
            mrr=mrr(retrieved_ids, gold_ids),
            first_gold_rank=rank,
            gold_count=len(gold_ids),
            sec_redundancy=section_redundancy(retrieved_ids, chunks_by_id, k),
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            latency_ms=latency,
        ))
        latencies.append(latency)

    n = len(results)
    ranked = sorted(r.first_gold_rank for r in results if r.first_gold_rank is not None)
    if ranked:
        mid = len(ranked) // 2
        median_rank = ranked[mid] if len(ranked) % 2 else (ranked[mid - 1] + ranked[mid]) / 2
        mean_rank = sum(ranked) / len(ranked)
    else:
        mean_rank = None
        median_rank = None
    return StrategyMetrics(
        name=name,
        hit1=sum(r.hit1 for r in results) / n,
        hit3=sum(r.hit3 for r in results) / n,
        hit5=sum(r.hit5 for r in results) / n,
        ndcg5=sum(r.ndcg5 for r in results) / n,
        evidence_recall5=sum(r.evidence_recall5 for r in results) / n,
        all_evidence_recall5=sum(r.all_evidence_recall5 for r in results) / n,
        mrr=sum(r.mrr for r in results) / n,
        mean_gold_rank=mean_rank,
        median_gold_rank=median_rank,
        latency_ms=sum(latencies) / len(latencies),
        results=results,
    )
