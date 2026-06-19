"""Library Mode index: HNSW + BM25 + CrossEncoder.

Loaded once at startup from pre-built index files.
Auto-selects strategy based on paper count:
  ≤ 500 papers → single-stage: BM25 + HNSW → weighted RRF → CE
  > 500 papers → two-stage:   BM25 paper-level max-pooling → within-paper CE
"""
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from paperpilot.core.classifier import is_yes_no_question
from paperpilot.core.router import route_cross_table
from paperpilot.core.indexer import (
    _deduplicate_sections, _merge_table_chunks,
    _WHY_HOW_TOP_K, _METHOD_BOOST, _METHOD_SECTION_RE,
    _chunk_has_number,
)

_DENSE_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
_CE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TWO_STAGE_THRESHOLD = 500    # papers
_RRF_K               = 60
_BM25_WEIGHT         = 1.0
_DENSE_WEIGHT        = 0.6   # reduced per benchmark: Dense cross-corpus no advantage
_POOL_SIZE           = 50
_TOP_PAPERS          = 8     # two-stage: how many papers to retrieve
_LOW_CONF_THRESHOLD  = 0.0
_MAX_SAME_SECTION    = 2
_YES_NO_TOP_K        = 8
_PAPER_QUERY_STOPWORDS = {
    "compare", "compared", "comparison", "different", "difference", "differences",
    "model", "models", "method", "methods", "paper", "papers", "approach",
    "approaches", "with", "between", "versus", "against", "better", "worse",
}

_dense_model: SentenceTransformer | None = None
_ce_model:    CrossEncoder | None        = None


def _load_models():
    global _dense_model, _ce_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(_DENSE_MODEL)
    if _ce_model is None:
        _ce_model = CrossEncoder(_CE_MODEL)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _paper_text(meta: dict) -> str:
    return " ".join([
        str(meta.get("title", "")),
        str(meta.get("abstract", "")),
    ])


def _paper_anchor_terms(query: str) -> set[str]:
    return {
        token for token in _tokenize(query)
        if len(token) >= 3 and token not in _PAPER_QUERY_STOPWORDS
    }


def _deduplicate_papers(results: list[dict], max_per_paper: int) -> list[dict]:
    counts: dict[str, int] = {}
    primary: list[dict] = []
    overflow: list[dict] = []
    for r in results:
        pid = r["chunk"].get("paper_id", "")
        count = counts.get(pid, 0)
        if count < max_per_paper:
            primary.append(r)
            counts[pid] = count + 1
        else:
            overflow.append(r)
    return primary + overflow


# ── LibraryIndex ──────────────────────────────────────────────────────────────

class LibraryIndex:
    """Manages BM25 + HNSW indexes for the pre-built paper library."""

    def __init__(self, index_dir: str | Path):
        import hnswlib

        index_dir = Path(index_dir)
        _load_models()

        # Load chunk registry
        with open(index_dir / "library_chunks.json", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        with open(index_dir / "library_papers.json", encoding="utf-8") as f:
            self.papers: dict = json.load(f)  # paper_id → {title, abstract, ...}

        with open(index_dir / "build_log.json", encoding="utf-8") as f:
            self.build_log: dict = json.load(f)

        self.chunks_by_id   = {c["chunk_id"]: c for c in self.chunks}
        self.chunk_ids      = [c["chunk_id"] for c in self.chunks]
        self.n_papers       = len(self.papers)

        # Paper → chunk indices (for two-stage scoped retrieval)
        self._paper_to_idxs: dict[str, list[int]] = defaultdict(list)
        for idx, c in enumerate(self.chunks):
            self._paper_to_idxs[c["paper_id"]].append(idx)

        # Full-corpus BM25
        corpus    = [_tokenize(c.get("index_text", c["text"])) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        # Paper-level BM25 over title + abstract. This is important for broad
        # library questions such as "compare different BERT models", where the
        # relevant unit is often the paper/model, not an arbitrary paragraph.
        self.paper_ids = list(self.papers.keys())
        self.paper_token_sets = {
            pid: set(_tokenize(_paper_text(self.papers[pid]))) for pid in self.paper_ids
        }
        self.paper_title_token_sets = {
            pid: set(_tokenize(str(self.papers[pid].get("title", ""))))
            for pid in self.paper_ids
        }
        paper_corpus = [list(self.paper_token_sets[pid]) for pid in self.paper_ids]
        self.paper_bm25 = BM25Okapi(paper_corpus)

        # Table-only BM25 for reliable numerical/comparison retrieval
        self.table_chunks = [c for c in self.chunks if c.get("chunk_type") == "table"]
        if self.table_chunks:
            table_texts     = [_tokenize(c.get("index_text", c["text"]))
                               for c in self.table_chunks]
            self.table_bm25 = BM25Okapi(table_texts)
        else:
            self.table_bm25 = None

        # HNSW (optional — falls back to numpy brute-force if hnswlib unavailable)
        try:
            import hnswlib
            self.hnsw = hnswlib.Index(space="cosine", dim=768)
            self.hnsw.load_index(str(index_dir / "library_hnsw.bin"),
                                 max_elements=len(self.chunks))
            self.hnsw.set_ef(_POOL_SIZE)
            self._embeddings = None
        except ImportError:
            self.hnsw = None
            emb_path = index_dir / "library_embeddings.npy"
            if emb_path.exists():
                self._embeddings = np.load(str(emb_path))
            else:
                raise RuntimeError(
                    "hnswlib not installed and library_embeddings.npy not found. "
                    "Install hnswlib or re-run build_library_index.py."
                )

    # ── Internal retrieval helpers ─────────────────────────────────────────

    def _table_bm25_top(self, query: str, k: int = 2) -> list[dict]:
        if not self.table_bm25:
            return []
        scores  = self.table_bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [{"chunk": self.table_chunks[i], "score": float(scores[i])}
                for i in top_idx]

    def _bm25_scores_all(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(_tokenize(query))

    def _hnsw_top(self, query: str, k: int) -> list[tuple[int, float]]:
        q_emb = _dense_model.encode([query], normalize_embeddings=True)[0]
        if self.hnsw is not None:
            labels, distances = self.hnsw.knn_query(q_emb.reshape(1, -1), k=k)
            return [(int(labels[0][i]), float(1 - distances[0][i]))
                    for i in range(len(labels[0]))]
        else:
            # Numpy fallback: brute-force cosine similarity
            scores  = self._embeddings @ q_emb
            top_idx = np.argsort(scores)[::-1][:k]
            return [(int(i), float(scores[i])) for i in top_idx]

    def _rrf_merge(self, bm25_scores: np.ndarray, hnsw_hits: list[tuple[int, float]],
                   pool: int) -> list[tuple[str, float]]:
        """Weighted RRF: BM25 weight=1.0, Dense weight=0.6 (per benchmark)."""
        rrf: dict[str, float] = {}

        bm25_top = np.argsort(bm25_scores)[::-1][:pool]
        for rank, idx in enumerate(bm25_top):
            cid = self.chunk_ids[idx]
            rrf[cid] = rrf.get(cid, 0) + _BM25_WEIGHT / (_RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(hnsw_hits):
            cid = self.chunk_ids[idx]
            rrf[cid] = rrf.get(cid, 0) + _DENSE_WEIGHT / (_RRF_K + rank + 1)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:pool]

    def _ce_rerank(self, query: str,
                   candidates: list[tuple[str, float]]) -> list[dict]:
        pairs     = [(query, self.chunks_by_id[cid]["text"]) for cid, _ in candidates]
        ce_scores = _ce_model.predict(pairs)
        reranked  = sorted(zip(candidates, ce_scores),
                           key=lambda x: x[1], reverse=True)
        return [{"chunk": self.chunks_by_id[cid], "score": float(s)}
                for (cid, _), s in reranked]

    def _postprocess(self, query: str, reranked: list[dict],
                     effective_k: int, llm_client=None,
                     query_type: str = "factual",
                     cross_table: bool | None = None) -> list[dict]:
        # Methodology boost before dedup
        if query_type == "methodology":
            for r in reranked:
                if _METHOD_SECTION_RE.search(r["chunk"].get("section", "")):
                    r["score"] += _METHOD_BOOST
            reranked.sort(key=lambda x: x["score"], reverse=True)

        results     = _deduplicate_sections(reranked, _MAX_SAME_SECTION)
        if query_type == "comparison":
            results = _deduplicate_papers(results, max_per_paper=1)
        if cross_table is None:
            cross_table = route_cross_table(query, llm_client)

        if cross_table:
            table_results = [r for r in results
                             if r["chunk"].get("chunk_type") == "table"]
            other_results = [r for r in results
                             if r["chunk"].get("chunk_type") != "table"]

            # Fallback: supplement with table BM25 if CE missed enough tables
            if len(table_results) < 2:
                seen_ids  = {r["chunk"]["chunk_id"] for r in table_results}
                bm25_tops = self._table_bm25_top(query, k=2)
                for t in bm25_tops:
                    if t["chunk"]["chunk_id"] not in seen_ids:
                        table_results.append(t)
                        seen_ids.add(t["chunk"]["chunk_id"])
                    if len(table_results) >= 2:
                        break

            if len(table_results) >= 2:
                composite = _merge_table_chunks(table_results)
                results   = [composite] + other_results[:effective_k - 1]
            else:
                results = results[:effective_k]
        else:
            results = results[:effective_k]

        # Numerical: inject top table via table BM25, bypassing CE Markdown scoring
        if query_type == "numerical" and not cross_table:
            has_table = any(r["chunk"].get("chunk_type") == "table" for r in results)
            if not has_table:
                bm25_tops = self._table_bm25_top(query, k=1)
                if bm25_tops:
                    results = [bm25_tops[0]] + results[:effective_k - 1]

        # Comparison: keep two-sided evidence and allow one table fallback even
        # when the query is comparative but not strictly cross-table.
        if query_type == "comparison" and not cross_table:
            has_table = any(r["chunk"].get("chunk_type") == "table" for r in results)
            if not has_table:
                bm25_tops = [r for r in self._table_bm25_top(query, k=1) if r["score"] > 0]
                if bm25_tops:
                    results = [bm25_tops[0]] + results[:effective_k - 1]
            if len(results) < 2:
                seen = {r["chunk"]["chunk_id"] for r in results}
                for r in reranked:
                    if r["chunk"]["chunk_id"] not in seen:
                        results.append(r)
                        break

        if results:
            low_confidence = results[0]["score"] < _LOW_CONF_THRESHOLD
            if query_type == "numerical":
                low_confidence = low_confidence or not any(
                    _chunk_has_number(r["chunk"]) for r in results
                )
            if query_type == "comparison":
                if cross_table:
                    table_like = [
                        r for r in results
                        if r["chunk"].get("chunk_type") in {"table", "merged_tables"}
                    ]
                    low_confidence = low_confidence or len(table_like) == 0
                low_confidence = low_confidence or len(results) < 2
            results[0]["low_confidence"] = low_confidence

        return results

    # ── Single-stage (≤ 500 papers) ────────────────────────────────────────

    def _search_single(self, query: str, effective_k: int,
                       llm_client=None, query_type: str = "factual",
                       cross_table: bool | None = None) -> list[dict]:
        bm25_scores = self._bm25_scores_all(query)
        hnsw_hits   = self._hnsw_top(query, _POOL_SIZE)
        candidates  = self._rrf_merge(bm25_scores, hnsw_hits, _POOL_SIZE)
        reranked    = self._ce_rerank(query, candidates)
        return self._postprocess(query, reranked, effective_k, llm_client,
                                 query_type, cross_table)

    # ── Two-stage (> 500 papers) ────────────────────────────────────────────

    def _search_two_stage(self, query: str, effective_k: int,
                          llm_client=None, query_type: str = "factual",
                          cross_table: bool | None = None) -> list[dict]:
        # Stage 1: paper-level BM25. Combine max-pooled chunk BM25 with
        # title/abstract BM25 so model-family queries can find the right papers
        # even when the best chunk is not yet in the top global chunk list.
        bm25_scores  = self._bm25_scores_all(query)
        paper_scores: dict[str, float] = {}
        for idx, score in enumerate(bm25_scores):
            pid = self.chunks[idx]["paper_id"]
            paper_scores[pid] = max(paper_scores.get(pid, 0.0), float(score))

        anchors = _paper_anchor_terms(query)
        paper_query_tokens = list(anchors) if anchors else _tokenize(query)
        paper_bm25_scores = self.paper_bm25.get_scores(paper_query_tokens)
        for idx, score in enumerate(paper_bm25_scores):
            pid = self.paper_ids[idx]
            if anchors:
                paper_scores[pid] = (0.25 * paper_scores.get(pid, 0.0)) + (2.0 * float(score))
                if anchors.intersection(self.paper_title_token_sets.get(pid, set())):
                    paper_scores[pid] += float(score)
            else:
                paper_scores[pid] = paper_scores.get(pid, 0.0) + float(score)

        if anchors:
            for pid in list(paper_scores):
                if not anchors.intersection(self.paper_token_sets.get(pid, set())):
                    paper_scores[pid] *= 0.1

        top_papers = sorted(paper_scores, key=lambda p: paper_scores[p],
                            reverse=True)[:_TOP_PAPERS]

        # Stage 2: within-paper CE for each top paper
        all_results: list[dict] = []
        for pid in top_papers:
            idxs        = self._paper_to_idxs[pid]
            paper_bm25  = [(self.chunk_ids[i], float(bm25_scores[i])) for i in idxs]
            paper_bm25.sort(key=lambda x: x[1], reverse=True)
            reranked    = self._ce_rerank(query, paper_bm25)
            all_results.extend(reranked[:effective_k])

        # Merge across papers by CE score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return self._postprocess(query, all_results, effective_k, llm_client,
                                 query_type, cross_table)

    # ── Public search API ───────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               llm_client=None, query_type: str = "factual",
               cross_table: bool | None = None) -> dict:
        """Search the library and return ranked results with metadata."""
        t0 = time.time()

        if is_yes_no_question(query):
            effective_k = max(top_k, _YES_NO_TOP_K)
        elif query_type == "why_how":
            effective_k = max(top_k, _WHY_HOW_TOP_K)
        else:
            effective_k = top_k

        strategy = "two_stage" if self.n_papers > _TWO_STAGE_THRESHOLD else "single_stage"

        if strategy == "two_stage":
            results = self._search_two_stage(query, effective_k, llm_client,
                                             query_type, cross_table)
        else:
            results = self._search_single(query, effective_k, llm_client,
                                          query_type, cross_table)

        latency_ms = (time.time() - t0) * 1000

        return {
            "strategy":   strategy,
            "results":    results,
            "latency_ms": latency_ms,
        }

    def paper_list(self) -> list[dict]:
        return [{"paper_id": pid, **meta}
                for pid, meta in self.papers.items()]

    def stats(self) -> dict:
        return {
            "n_papers": self.n_papers,
            "n_chunks": len(self.chunks),
            **self.build_log,
        }
