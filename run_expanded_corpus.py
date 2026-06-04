"""Expanded-corpus cross-paper retrieval experiment.

Adds QASPER train split papers (~888 papers, ~47k chunks) to the retrieval
corpus as distractors.  Dev queries and qrels remain unchanged.

Goal: test whether Dense gains an advantage over BM25 at 6x corpus scale,
and how all strategies degrade as the haystack grows.

Usage:
    python run_expanded_corpus.py          # full run
    python run_expanded_corpus.py --dry-run  # skip Dense encoding, use existing cache
"""
import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from bench.config import (
    CE_MODEL, DENSE_MODEL, RRF_K, TOP_K, POOL_SIZE, EMBEDDING_CACHE,
)
from bench.dataset import load_qasper, filter_papers, extract_answerable_queries
from bench.chunking import build_chunks, build_qrels, filter_noise_chunks
from bench.index import BM25Index, DenseIndex, tokenize
from bench.retrievers import retrieve_bm25, retrieve_dense, retrieve_rrf, CERetriever
from bench.query_classifier import classify_all
from bench.metrics import ndcg_at_k, recall_at_k, mrr, evaluate_strategy

TRAIN_EMB_CACHE    = "data/embeddings_mpnet_train.npy"
EXPANDED_EMB_CACHE = "data/embeddings_mpnet_expanded.npy"
RESULTS_OUT        = "data/expanded_corpus_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_train_papers():
    """Load QASPER train split from local JSON (already downloaded)."""
    import os, json
    path = "data/qasper_raw/qasper-train-v0.3.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train JSON not found at {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Convert to list-of-dicts format matching datasets API
    papers = []
    for arxiv_id, data in raw.items():
        papers.append({
            "id": arxiv_id,
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "full_text": data.get("full_text", []),
            "qas": data.get("qas", []),
        })
    return papers


def build_train_chunks(train_papers, min_chars=80):
    """Build chunks from train papers, apply length filter (no qrel needed)."""
    chunks = []
    for paper in train_papers:
        paper_id = paper["id"]
        for sec_idx, section_data in enumerate(paper["full_text"]):
            section = section_data.get("section_name", "")
            for para_idx, para_text in enumerate(section_data.get("paragraphs", [])):
                if not para_text.strip() or len(para_text) < min_chars:
                    continue
                chunks.append({
                    "chunk_id": f"{paper_id}__s{sec_idx}__p{para_idx}",
                    "paper_id": paper_id,
                    "section":  section,
                    "sec_idx":  sec_idx,
                    "text":     para_text,
                })
    return chunks


def encode_train_chunks(train_chunks, dense_index: DenseIndex):
    """Encode train chunks using already-loaded Dense model."""
    import os
    if os.path.exists(TRAIN_EMB_CACHE):
        print(f"  Loading train embeddings from cache: {TRAIN_EMB_CACHE}")
        return np.load(TRAIN_EMB_CACHE)

    print(f"  Encoding {len(train_chunks)} train chunks (this takes ~15-25 min)...")
    texts = [c["text"] for c in train_chunks]
    embs  = dense_index.model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(TRAIN_EMB_CACHE, embs)
    print(f"  Saved → {TRAIN_EMB_CACHE}")
    return embs


def expected_ndcg_random(n_pool, n_gold, k=5):
    w    = sum(1 / math.log2(i + 2) for i in range(k))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(int(round(n_gold)), k)))
    return (n_gold / n_pool) * w / idcg if idcg > 0 else 0.0


# ── Expanded index wrappers ───────────────────────────────────────────────────

class ExpandedDenseIndex:
    """Wraps pre-computed dev + train embeddings into one retrieval surface."""
    def __init__(self, all_chunks, expanded_embeddings, base_dense: DenseIndex):
        self.chunk_ids  = [c["chunk_id"] for c in all_chunks]
        self.embeddings = expanded_embeddings
        self._model     = base_dense

    def encode_query(self, query_text):
        return self._model.encode_query(query_text)

    def get_scores(self, query_text):
        return self.embeddings @ self.encode_query(query_text)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run=False):

    # ── Dev data (queries + qrels unchanged) ─────────────────────────────────
    print("=== Loading dev data ===")
    ds         = load_qasper()
    dev_papers = filter_papers(ds)
    queries    = extract_answerable_queries(dev_papers)
    dev_chunks = build_chunks(dev_papers)
    qrels      = build_qrels(queries, dev_chunks)
    dev_chunks = filter_noise_chunks(dev_chunks, qrels)
    print(f"Dev: {len(dev_papers)} papers, {len(dev_chunks)} chunks, {len(qrels)} queries")

    query_type_map = classify_all(queries)
    for q in queries:
        q["query_type"] = query_type_map.get(q["qid"], "factual")

    # ── Train data (distractors only) ─────────────────────────────────────────
    print("\n=== Loading train papers as distractors ===")
    train_papers = load_train_papers()
    train_chunks = build_train_chunks(train_papers)
    print(f"Train: {len(train_papers)} papers, {len(train_chunks)} chunks")

    # ── Combined corpus ───────────────────────────────────────────────────────
    all_chunks   = dev_chunks + train_chunks
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
    print(f"\nExpanded corpus: {len(all_chunks)} chunks total")
    print(f"  dev gold chunks ÷ total = {sum(len(v) for v in qrels.values())} / {len(all_chunks)}")

    # ── Indexes ───────────────────────────────────────────────────────────────
    print("\n=== Building BM25 index (expanded) ===")
    bm25_index = BM25Index(all_chunks)

    print("\n=== Building Dense index (expanded) ===")
    # The embedding cache was built before noise filtering, so we must
    # re-align it to the filtered dev_chunks by chunk_id lookup.
    unfiltered_dev = build_chunks(dev_papers)          # pre-filter, matches cache
    base_dense     = DenseIndex(unfiltered_dev)        # loads cache (11k rows)
    id_to_cache    = {cid: i for i, cid in enumerate(base_dense.chunk_ids)}
    dev_indices    = [id_to_cache[c["chunk_id"]] for c in dev_chunks]
    dev_embs       = base_dense.embeddings[dev_indices]  # (10307, 768)

    if dry_run:
        print("  [dry-run] Skipping train encoding, Dense experiment omitted.")
        train_embs = None
    else:
        train_embs = encode_train_chunks(train_chunks, base_dense)

    if train_embs is not None:
        import os
        if os.path.exists(EXPANDED_EMB_CACHE):
            print(f"  Loading expanded embeddings from cache: {EXPANDED_EMB_CACHE}")
            expanded_embs = np.load(EXPANDED_EMB_CACHE)
        else:
            expanded_embs = np.concatenate([dev_embs, train_embs], axis=0)
            np.save(EXPANDED_EMB_CACHE, expanded_embs)
            print(f"  Saved expanded embeddings → {EXPANDED_EMB_CACHE}")

        exp_dense = ExpandedDenseIndex(all_chunks, expanded_embs, base_dense)
    else:
        # dry-run: Dense uses dev-only aligned embeddings via ExpandedDenseIndex
        exp_dense = ExpandedDenseIndex(dev_chunks, dev_embs, base_dense)

    print("\n=== Loading CrossEncoder ===")
    ce_retriever = CERetriever()

    # ── Strategy closures ─────────────────────────────────────────────────────
    strategies = {
        "BM25": lambda q: retrieve_bm25(q, bm25_index),
        "RRF":  lambda q: retrieve_rrf(q, bm25_index,
                                       exp_dense if exp_dense else base_dense),
        "CE":   lambda q: ce_retriever.retrieve(q, bm25_index,
                                                exp_dense if exp_dense else base_dense,
                                                chunks_by_id),
    }
    if exp_dense:
        strategies["Dense"] = lambda q: retrieve_dense(q, exp_dense)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    valid_queries = [q for q in queries if q["qid"] in qrels]
    n_corpus      = len(all_chunks)
    n_gold_avg    = sum(len(v) for v in qrels.values()) / len(qrels)
    random_base   = expected_ndcg_random(n_corpus, n_gold_avg)

    print(f"\n=== Evaluating {len(valid_queries)} queries on {n_corpus}-chunk corpus ===")
    print(f"Random baseline NDCG@5 = {random_base:.6f}\n")

    CROSS_DEV = {"BM25": 0.1023, "Dense": 0.1002, "RRF": 0.1287, "CE": 0.1718}
    results   = {}

    for name, fn in strategies.items():
        print(f"  {name}...")
        t0  = time.time()
        sm  = evaluate_strategy(fn, valid_queries, qrels, chunks_by_id,
                                 query_type_map, name=name)
        elapsed = time.time() - t0
        results[name] = {
            "ndcg5": sm.ndcg5, "recall5": sm.recall5,
            "mrr": sm.mrr,     "latency_ms": sm.latency_ms,
        }
        delta = sm.ndcg5 - CROSS_DEV.get(name, sm.ndcg5)
        print(f"    NDCG@5={sm.ndcg5:.4f}  Recall@5={sm.recall5:.4f}  "
              f"MRR={sm.mrr:.4f}  Lat={sm.latency_ms:.1f}ms  "
              f"vs_dev_only={delta:+.4f}  ({elapsed:.0f}s total)")

    # ── Results table ─────────────────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("EXPANDED CORPUS — Tier 1 Summary")
    print(f"Corpus: {n_corpus} chunks  |  Random NDCG@5: {random_base:.5f}")
    print(SEP)
    print(f"{'Strategy':<8} {'NDCG@5':>8} {'Recall@5':>10} {'MRR':>8} "
          f"{'Lat(ms)':>9}  dev_only_NDCG  delta")
    print("-" * 72)
    for name in ["BM25", "Dense", "RRF", "CE"]:
        if name not in results:
            print(f"{name:<8}  — (skipped in dry-run)")
            continue
        r     = results[name]
        cross = CROSS_DEV.get(name, 0)
        delta = r["ndcg5"] - cross
        print(f"{name:<8}  {r['ndcg5']:.4f}  {r['recall5']:>10.4f}  "
              f"{r['mrr']:>8.4f}  {r['latency_ms']:>9.1f}  "
              f"{cross:.4f}          {delta:+.4f}")

    # ── Per query-type (reuse strategy metrics from Tier 1 run) ──────────────
    print(f"\n{SEP}")
    print("EXPANDED CORPUS — NDCG@5 by Query Type")
    print(SEP)
    CROSS_T15 = {
        "factual":     {"BM25": 0.0977, "Dense": 0.0905, "RRF": 0.1177, "CE": 0.1492},
        "why_how":     {"BM25": 0.1345, "Dense": 0.1213, "RRF": 0.1520, "CE": 0.2433},
        "numerical":   {"BM25": 0.1236, "Dense": 0.2166, "RRF": 0.2260, "CE": 0.3152},
        "methodology": {"BM25": 0.0785, "Dense": 0.0852, "RRF": 0.1145, "CE": 0.1253},
        "comparison":  {"BM25": 0.0481, "Dense": 0.0303, "RRF": 0.0884, "CE": 0.0841},
    }

    type_results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for name, fn in strategies.items():
        sm = evaluate_strategy(fn, valid_queries, qrels, chunks_by_id,
                               query_type_map, name=name)
        for r in sm.results:
            type_results[r.query_type][name].append(r.ndcg5)

    strat_keys = [s for s in ["BM25", "Dense", "RRF", "CE"] if s in strategies]
    hdr = f"{'Type':<14}" + "".join(f"  {s:>7}" for s in strat_keys) + "   (dev_only CE)"
    print(hdr)
    print("-" * 64)
    t15: dict = {}
    for qtype in sorted(CROSS_T15):
        vals = {s: float(np.mean(type_results[qtype][s]))
                for s in strat_keys if type_results[qtype].get(s)}
        t15[qtype] = vals
        n    = len(type_results[qtype].get("BM25", []))
        line = f"{qtype:<10}(N={n:<3})"
        for s in strat_keys:
            line += f"  {vals.get(s, 0):.4f}"
        line += f"   {CROSS_T15[qtype].get('CE', 0):.4f}"
        print(line)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "corpus_size":    n_corpus,
        "dev_only_size":  len(dev_chunks),
        "train_size":     len(train_chunks),
        "random_baseline": random_base,
        "tier1":   results,
        "tier1_dev_only": CROSS_DEV,
        "tier15":  t15,
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {RESULTS_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Dense train encoding (BM25/RRF/CE only)")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
