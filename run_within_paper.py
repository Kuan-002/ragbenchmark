"""Within-paper retrieval — full Tier 1 / 1.5 / 1.6 / 2 / 3 experiment.

Each query is evaluated against only the chunks from its own paper,
mirroring the production web-app scenario (single uploaded paper).

Usage:
  # Without LLM (Tier 1 / 1.5 / 1.6 only, fast)
  python run_within_paper.py --no-llm

  # With LLM (all tiers, needs OPENAI_API_KEY or DEEPSEEK_API_KEY in .env)
  python run_within_paper.py --llm-per-type 10
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from bench.config import CE_MODEL, RRF_K, TOP_K, POOL_SIZE, LLM_PER_TYPE, BADCASE_PARTIAL_N
from bench.dataset import load_qasper, filter_papers, extract_answerable_queries
from bench.chunking import build_chunks, build_qrels, filter_noise_chunks
from bench.index import DenseIndex, tokenize
from bench.metrics import (
    QueryResult, ndcg_at_k, recall_at_k, mrr, section_redundancy,
)
from bench.query_classifier import classify_all
from bench.llm_judge import stratified_sample, run_two_pass_judge


# ── Per-paper index wrappers ──────────────────────────────────────────────────

class _PaperBM25:
    def __init__(self, paper_chunks):
        self.chunk_ids = [c["chunk_id"] for c in paper_chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in paper_chunks])

    def get_scores(self, query_text):
        return self.bm25.get_scores(tokenize(query_text))


class _PaperDense:
    """Slice of pre-computed embeddings — no re-encoding needed."""
    def __init__(self, paper_chunks, full_dense: DenseIndex):
        self.chunk_ids = [c["chunk_id"] for c in paper_chunks]
        id_to_idx      = {cid: i for i, cid in enumerate(full_dense.chunk_ids)}
        indices        = [id_to_idx[cid] for cid in self.chunk_ids]
        self.embeddings = full_dense.embeddings[indices]
        self._model     = full_dense

    def get_scores(self, query_text):
        return self.embeddings @ self._model.encode_query(query_text)


# ── Within-paper retrieval helpers ───────────────────────────────────────────

def _topk(scores, chunk_ids, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(chunk_ids[i], float(scores[i])) for i in idx]


def _wp_bm25(q, bm25, k=TOP_K):
    return _topk(bm25.get_scores(q), bm25.chunk_ids, k)


def _wp_dense(q, dense, k=TOP_K):
    return _topk(dense.get_scores(q), dense.chunk_ids, k)


def _wp_rrf(q, bm25, dense, k=TOP_K):
    n = len(bm25.chunk_ids)
    b_top = np.argsort(bm25.get_scores(q))[::-1][:n]
    d_top = np.argsort(dense.get_scores(q))[::-1][:n]
    scores: dict[str, float] = {}
    for rank, i in enumerate(b_top):
        cid = bm25.chunk_ids[i]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    for rank, i in enumerate(d_top):
        cid = dense.chunk_ids[i]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def _wp_ce(q, bm25, dense, chunks_by_id, ce_model, k=TOP_K):
    n     = len(bm25.chunk_ids)
    pool  = min(POOL_SIZE, n)
    cands = _wp_rrf(q, bm25, dense, k=pool)
    pairs = [(q, chunks_by_id[cid]["text"]) for cid, _ in cands]
    ces   = ce_model.predict(pairs)
    reranked = sorted(zip([c for c, _ in cands], ces),
                      key=lambda x: x[1], reverse=True)
    return [(cid, float(s)) for cid, s in reranked[:k]]


# ── Random baseline ───────────────────────────────────────────────────────────

def _expected_ndcg(n_pool, n_gold, k=5):
    w    = sum(1 / math.log2(i + 2) for i in range(k))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return (n_gold / n_pool) * w / idcg if idcg > 0 else 0.0


# ── Print helpers ─────────────────────────────────────────────────────────────

STRATEGIES = ["BM25", "Dense", "RRF", "CE"]
SEP = "=" * 72


def _header(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(no_llm=False, llm_client=None, llm_per_type=LLM_PER_TYPE):

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    ds      = load_qasper()
    papers  = filter_papers(ds)
    queries = extract_answerable_queries(papers)
    chunks  = build_chunks(papers)
    qrels   = build_qrels(queries, chunks)
    chunks  = filter_noise_chunks(chunks, qrels)

    chunks_by_id   = {c["chunk_id"]: c for c in chunks}
    query_type_map = classify_all(queries)
    for q in queries:
        q["query_type"] = query_type_map.get(q["qid"], "factual")

    paper_chunks: dict[str, list] = defaultdict(list)
    for c in chunks:
        paper_chunks[c["paper_id"]].append(c)

    # ── Indexes ───────────────────────────────────────────────────────────────
    print("Loading Dense index from cache...")
    full_dense = DenseIndex(chunks)
    print("Loading CrossEncoder...")
    ce_model = CrossEncoder(CE_MODEL)

    print("Building per-paper BM25 & Dense indexes...")
    paper_bm25  = {pid: _PaperBM25(pc)  for pid, pc in paper_chunks.items()}
    paper_dense = {pid: _PaperDense(pc, full_dense) for pid, pc in paper_chunks.items()}

    # ── Evaluation loop ───────────────────────────────────────────────────────
    valid_queries = [q for q in queries if q["qid"] in qrels]
    print(f"Evaluating {len(valid_queries)} queries across all strategies...\n")

    all_results: dict[str, list[QueryResult]] = {s: [] for s in STRATEGIES}
    random_baselines = []

    for query in valid_queries:
        qid      = query["qid"]
        pid      = query["paper_id"]
        gold_ids = list(qrels[qid])
        bm25_idx = paper_bm25[pid]
        dens_idx = paper_dense[pid]
        q_text   = query["question"]
        n_pool   = len(paper_chunks[pid])

        random_baselines.append(_expected_ndcg(n_pool, len(gold_ids)))

        fns = {
            "BM25":  lambda q: _wp_bm25(q, bm25_idx),
            "Dense": lambda q: _wp_dense(q, dens_idx),
            "RRF":   lambda q: _wp_rrf(q, bm25_idx, dens_idx),
            "CE":    lambda q: _wp_ce(q, bm25_idx, dens_idx, chunks_by_id, ce_model),
        }

        for name, fn in fns.items():
            t0      = time.time()
            ret     = fn(q_text)
            lat     = (time.time() - t0) * 1000
            ret_ids = [cid for cid, _ in ret]

            all_results[name].append(QueryResult(
                qid=qid,
                question=q_text,
                query_type=query["query_type"],
                answer_type=query.get("answer_type"),
                nlp_background=query.get("nlp_background"),
                topic_background=query.get("topic_background"),
                ndcg5=ndcg_at_k(ret_ids, gold_ids),
                recall5=recall_at_k(ret_ids, gold_ids),
                mrr=mrr(ret_ids, gold_ids),
                sec_redundancy=section_redundancy(ret_ids, chunks_by_id),
                retrieved_ids=ret_ids,
                gold_ids=gold_ids,
                latency_ms=lat,
            ))

    random_base = float(np.mean(random_baselines))

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    _header("TIER 1 — Within-Paper Retrieval Metrics")
    CROSS = {"BM25": 0.1023, "Dense": 0.1002, "RRF": 0.1287, "CE": 0.1718}

    print(f"{'Strategy':<8} {'NDCG@5':>8} {'Recall@5':>10} {'MRR':>8} "
          f"{'SecRed':>8} {'Lat(ms)':>9}  lift_vs_random  cross_NDCG")
    print("-" * 72)
    print(f"{'Random':<8}  {random_base:.4f}  {'—':>10}  {'—':>8}  "
          f"{'—':>8}  {'—':>9}")
    print("-" * 72)

    t1: dict[str, dict] = {}
    for name in STRATEGIES:
        rs   = all_results[name]
        ndcg = np.mean([r.ndcg5   for r in rs])
        rec  = np.mean([r.recall5 for r in rs])
        m    = np.mean([r.mrr     for r in rs])
        red  = np.mean([r.sec_redundancy for r in rs])
        lat  = np.mean([r.latency_ms     for r in rs])
        lift = (ndcg - random_base) / random_base * 100
        t1[name] = {"ndcg5": ndcg, "recall5": rec, "mrr": m,
                    "sec_redundancy": red, "latency_ms": lat}
        print(f"{name:<8}  {ndcg:.4f}  {rec:>10.4f}  {m:>8.4f}  {red:>8.4f}  "
              f"{lat:>9.1f}  {lift:+.0f}%"
              f"{'':>10}{CROSS.get(name, 0):.4f}")

    # ── Tier 1.5 ──────────────────────────────────────────────────────────────
    _header("TIER 1.5 — Within-Paper NDCG@5 by Query Type")
    type_counts: dict[str, int] = defaultdict(int)
    for r in all_results["CE"]:
        type_counts[r.query_type] += 1

    hdr = f"{'Type':<14}" + "".join(f"  {s:>7}" for s in STRATEGIES)
    print(hdr)
    print("-" * 56)
    t15: dict = {}
    for qtype in sorted(type_counts):
        n    = type_counts[qtype]
        line = f"{qtype:<10}(N={n:<3})"
        t15[qtype] = {}
        for name in STRATEGIES:
            vals = [r.ndcg5 for r in all_results[name] if r.query_type == qtype]
            v    = float(np.mean(vals))
            t15[qtype][name] = v
            line += f"  {v:.4f}"
        print(line)

    # ── Tier 1.6 ──────────────────────────────────────────────────────────────
    _header("TIER 1.6 — Within-Paper CE Breakdown")

    def _breakdown(field, rs):
        by: dict[str, list] = defaultdict(list)
        for r in rs:
            by[getattr(r, field) or "unknown"].append(r.ndcg5)
        return {k: (len(v), float(np.mean(v))) for k, v in by.items()}

    ce_rs = all_results["CE"]
    for field in ("answer_type", "nlp_background", "topic_background"):
        print(f"\n  [{field}]")
        for k, (n, v) in sorted(_breakdown(field, ce_rs).items(),
                                 key=lambda x: -x[1][1]):
            print(f"    {k:<20}  N={n:<4}  NDCG@5={v:.4f}")

    # ── Tier 2 — LLM Sufficiency ─────────────────────────────────────────────
    sufficiency_maps = {s: {} for s in STRATEGIES}
    badcase_pools    = {s: [] for s in STRATEGIES}

    if not no_llm and llm_client is not None:
        _header("TIER 2 — LLM Sufficiency (within-paper CE sample)")

        # Stratify on CE results (same qids used for all strategies)
        sampled_ce   = stratified_sample(all_results["CE"], n_per_type=llm_per_type)
        sampled_qids = {r.qid for r in sampled_ce}
        print(f"Sampled {len(sampled_qids)} queries (stratified by type)\n")

        for name in STRATEGIES:
            print(f"  Strategy: {name}")
            by_qid     = {r.qid: r for r in all_results[name]}
            sampled_s  = [by_qid[q] for q in sampled_qids if q in by_qid]
            suf_map, bcases = run_two_pass_judge(
                all_results[name], sampled_s, chunks_by_id, llm_client,
                badcase_partial_n=BADCASE_PARTIAL_N,
            )
            sufficiency_maps[name] = suf_map
            badcase_pools[name]    = bcases

        # Print Tier 2 table
        print(f"\n{'Strategy':<8} {'Fully':>8} {'Partial':>9} {'Insuf':>8}")
        print("-" * 40)
        t2: dict = {}
        for name in STRATEGIES:
            sm   = sufficiency_maps[name]
            n    = len(sm)
            full = sum(1 for v in sm.values() if v == "FULLY_SUFFICIENT")
            part = sum(1 for v in sm.values() if v == "PARTIALLY_SUFFICIENT")
            ins  = sum(1 for v in sm.values() if v == "INSUFFICIENT")
            t2[name] = {"n": n, "full": full, "partial": part, "insufficient": ins}
            print(f"{name:<8}  {full/n*100:6.0f}% ({full}/{n})"
                  f"  {part/n*100:5.0f}% ({part}/{n})"
                  f"  {ins/n*100:5.0f}% ({ins}/{n})")

        # ── Tier 3 ────────────────────────────────────────────────────────────
        _header("TIER 3 — Badcase Root-Cause (within-paper)")
        print(f"{'Strategy':<8} {'N':>4} {'Fixable':>9} "
              f"{'RET_FAIL':>10} {'MISS_EV':>9} {'REDUND':>8}")
        print("-" * 56)
        t3: dict = {}
        for name in STRATEGIES:
            pool = badcase_pools[name]
            if not pool:
                print(f"{name:<8}  — (no badcases)")
                continue
            cats = [r.badcase_category for r in pool if r.badcase_category]
            n    = len(cats)
            rf   = cats.count("RETRIEVAL_FAILURE")
            me   = cats.count("MISSING_EVIDENCE")
            rd   = cats.count("REDUNDANCY")
            fix  = rf + rd
            t3[name] = {"n": n, "fixable": fix, "retrieval_failure": rf,
                        "missing_evidence": me, "redundancy": rd}
            print(f"{name:<8}  {n:>4}  {fix/n*100:6.0f}% ({fix}/{n})"
                  f"  {rf/n*100:6.0f}% ({rf}/{n})"
                  f"  {me/n*100:5.0f}% ({me}/{n})"
                  f"  {rd/n*100:5.0f}% ({rd}/{n})")
    else:
        print("\n[Tier 2 & 3 skipped — no LLM client]")
        t2, t3 = {}, {}

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "within_paper_random_baseline": random_base,
        "cross_corpus_random_baseline": 0.0003,
        "tier1":  t1,
        "tier15": t15,
        "tier2":  t2,
        "tier3":  t3,
        "cross_corpus_tier1": CROSS,
    }
    with open("data/within_paper_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → data/within_paper_results.json")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Tier 2 & 3 LLM judge")
    parser.add_argument("--llm-per-type", type=int, default=LLM_PER_TYPE,
                        help="LLM judge samples per query type (default 10)")
    args = parser.parse_args()

    llm_client = None
    if not args.no_llm:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            import openai
            base_url = os.environ.get("OPENAI_BASE_URL") or (
                "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
            )
            llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            print(f"LLM client: model={model}, base_url={base_url}")
        else:
            print("No API key found — running without LLM (Tier 1/1.5/1.6 only).")
            args.no_llm = True

    run(no_llm=args.no_llm, llm_client=llm_client,
        llm_per_type=args.llm_per_type)


if __name__ == "__main__":
    main()
