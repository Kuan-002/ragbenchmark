"""Build BM25 index + CrossEncoder reranker for a single paper.

Benchmark conclusions applied:
  - Dense within-paper ≈ random baseline (+5%) → removed entirely
  - BM25 → CE is the optimal pipeline for single-paper retrieval

Post-retrieval rules:
  - Section deduplication: max 2 chunks per section
  - Yes/No expansion: top_k enlarged to 8
  - Low-confidence flag: CE top score < 0

Table-specific improvements:
  - BM25 uses index_text (flat caption + cell values) for tables,
    so structural Markdown characters don't pollute token frequencies
  - Cross-table queries: top-2 table chunks are force-included and merged
    into a composite chunk so the LLM can compare them jointly

Query-type routing (driven by classifier output):
  - numerical:   force-inject via table-BM25 (index_text exact match on caption
                 + cell values); bypasses CE's unreliable Markdown scoring
  - comparison:  same table-BM25 fallback when CE surfaces < 2 tables
  - why_how:     expand top_k to 7 for broader multi-paragraph coverage
  - methodology: boost chunks whose section name matches method/architecture
                 keywords before section deduplication
"""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from paperpilot.core.classifier import is_yes_no_question

_CE_MODEL           = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_LOW_CONF_THRESHOLD = 0.0
_MAX_SAME_SECTION   = 2
_YES_NO_TOP_K       = 8
_WHY_HOW_TOP_K      = 7
_METHOD_BOOST       = 0.5
_MAX_MERGED_TABLES  = 2   # how many tables to merge for cross-table queries

_METHOD_SECTION_RE = re.compile(
    r"(?i)\b(method|model|approach|architect|train|experiment|setup|implement)\b"
)

_ce_model: CrossEncoder | None = None


def load_models():
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder(_CE_MODEL)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _deduplicate_sections(results: list[dict], max_per_section: int) -> list[dict]:
    counts:   dict[str, int] = {}
    primary:  list[dict]     = []
    overflow: list[dict]     = []
    for r in results:
        sec   = r["chunk"].get("section", "")
        count = counts.get(sec, 0)
        if count < max_per_section:
            primary.append(r)
            counts[sec] = count + 1
        else:
            overflow.append(r)
    return primary + overflow


def _merge_table_chunks(table_results: list[dict]) -> dict:
    """Combine multiple table chunks into one composite chunk for joint LLM reasoning."""
    tables = table_results[:_MAX_MERGED_TABLES]
    merged_text = "\n\n---\n\n".join(t["chunk"]["text"] for t in tables)
    captions    = " vs. ".join(
        t["chunk"].get("caption", "Table") for t in tables
    )
    return {
        "chunk": {
            "chunk_id":   "merged_tables",
            "section":    "Tables",
            "text":       merged_text,
            "chunk_type": "merged_tables",
            "caption":    captions,
        },
        "score": tables[0]["score"],  # inherit top table's score
    }


def _chunk_has_number(chunk: dict) -> bool:
    return bool(re.search(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?", chunk.get("text", "")))


class PaperIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks       = chunks
        self.chunks_by_id = {c["chunk_id"]: c for c in chunks}
        self.chunk_ids    = [c["chunk_id"] for c in chunks]

        # Full-corpus BM25: uses index_text for tables so Markdown pipes
        # don't pollute term frequencies
        index_texts = [_tokenize(c.get("index_text", c["text"])) for c in chunks]
        self.bm25   = BM25Okapi(index_texts)

        # Table-only BM25: built on index_text (caption + deduplicated cell
        # values) so numerical/comparison queries can keyword-match tables
        # directly without depending on CE's unreliable Markdown scoring
        self.table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
        if self.table_chunks:
            table_texts      = [_tokenize(c.get("index_text", c["text"]))
                                for c in self.table_chunks]
            self.table_bm25  = BM25Okapi(table_texts)
        else:
            self.table_bm25  = None

    def _table_bm25_top(self, query: str, k: int = 2) -> list[dict]:
        """Return top-k table chunks ranked by table-only BM25 (index_text)."""
        if not self.table_bm25:
            return []
        scores  = self.table_bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [{"chunk": self.table_chunks[i], "score": float(scores[i])}
                for i in top_idx]

    def _bm25_top(self, query: str, k: int) -> list[dict]:
        scores  = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {"chunk": self.chunks_by_id[self.chunk_ids[i]], "score": float(scores[i])}
            for i in top_idx
        ]

    def retrieve_rrf(self, query: str, top_k: int = 5, **_kwargs) -> list[dict]:
        """Pure BM25 retrieval (Dense removed per benchmark findings)."""
        return self._bm25_top(query, top_k)

    def retrieve_ce(self, query: str, top_k: int = 5,
                    pool_size: int | None = None,
                    cross_table: bool = False,
                    query_type: str = "factual") -> list[dict]:
        """BM25 pool → CE rerank → post-processing.

        Args:
            cross_table: Pre-computed routing decision from router.route_cross_table().
                         When True, top-2 table chunks are merged into a composite
                         chunk for joint LLM reasoning.
            query_type:  Classifier output driving retrieval behaviour:
                         - numerical   → force-inject top table chunk if absent
                         - why_how     → expand effective_k to _WHY_HOW_TOP_K
                         - methodology → boost section-matched chunks before dedup

        Post-processing order:
          1. Effective k expansion (yes/no → 8, why_how → 7)
          2. CE reranking
          3. Methodology section boost
          4. Section deduplication
          5. Cross-table merging / numerical table injection
          6. Low-confidence flag
        """
        n = len(self.chunks)

        # 1. Effective k expansion
        if is_yes_no_question(query):
            effective_k = max(top_k, _YES_NO_TOP_K)
        elif query_type == "why_how":
            effective_k = max(top_k, _WHY_HOW_TOP_K)
        else:
            effective_k = top_k

        # BM25 candidate pool
        pool       = min(pool_size or n, n)
        candidates = self._bm25_top(query, pool)

        # CE reranking
        pairs     = [(query, c["chunk"]["text"]) for c in candidates]
        ce_scores = _ce_model.predict(pairs)

        reranked    = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
        all_results = [{"chunk": c["chunk"], "score": float(s)} for c, s in reranked]
        results     = list(all_results)

        # 2. Methodology boost: re-score before dedup so boosted chunks survive
        if query_type == "methodology":
            for r in results:
                if _METHOD_SECTION_RE.search(r["chunk"].get("section", "")):
                    r["score"] += _METHOD_BOOST
            results.sort(key=lambda x: x["score"], reverse=True)

        # 3. Section deduplication
        results = _deduplicate_sections(results, _MAX_SAME_SECTION)

        # 4a. Cross-table merging (routing decision injected by caller)
        if cross_table:
            table_results = [r for r in results
                             if r["chunk"].get("chunk_type") == "table"]
            other_results = [r for r in results
                             if r["chunk"].get("chunk_type") != "table"]

            # Fallback: if CE didn't surface enough tables, use table BM25
            if len(table_results) < 2:
                seen_ids   = {r["chunk"]["chunk_id"] for r in table_results}
                bm25_tops  = self._table_bm25_top(query, k=2)
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

        # 4b. Numerical: force top table chunk via table BM25 (index_text
        #     exact match), bypassing CE's unreliable Markdown scoring
        if query_type == "numerical" and not cross_table:
            has_table = any(r["chunk"].get("chunk_type") == "table" for r in results)
            if not has_table:
                bm25_tops = self._table_bm25_top(query, k=1)
                if bm25_tops:
                    results = [bm25_tops[0]] + results[:effective_k - 1]

        # 4c. Comparison: keep at least two evidence chunks and give table
        #     evidence a chance even when the query is not strictly cross-table.
        if query_type == "comparison" and not cross_table:
            has_table = any(r["chunk"].get("chunk_type") == "table" for r in results)
            if not has_table:
                bm25_tops = [r for r in self._table_bm25_top(query, k=1) if r["score"] > 0]
                if bm25_tops:
                    results = [bm25_tops[0]] + results[:effective_k - 1]
            if len(results) < 2:
                seen = {r["chunk"]["chunk_id"] for r in results}
                for r in all_results:
                    if r["chunk"]["chunk_id"] not in seen:
                        results.append(r)
                        break

        # 5. Low-confidence flag
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
