import numpy as np
from sentence_transformers import CrossEncoder
from bench.config import CE_MODEL, RRF_K, CANDIDATE_K, POOL_SIZE, TOP_K
from bench.index import tokenize


def retrieve_bm25(query_text, bm25_index, top_k=TOP_K):
    scores = bm25_index.get_scores(query_text)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(bm25_index.chunk_ids[i], float(scores[i])) for i in top_indices]


def retrieve_dense(query_text, dense_index, top_k=TOP_K):
    scores = dense_index.get_scores(query_text)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(dense_index.chunk_ids[i], float(scores[i])) for i in top_indices]


def retrieve_rrf(query_text, bm25_index, dense_index,
                 top_k=TOP_K, rrf_k=RRF_K, candidate_k=CANDIDATE_K):
    bm25_results  = retrieve_bm25(query_text, bm25_index, top_k=candidate_k)
    dense_results = retrieve_dense(query_text, dense_index, top_k=candidate_k)

    rrf_scores = {}
    for rank, (cid, _) in enumerate(bm25_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
    for rank, (cid, _) in enumerate(dense_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class CERetriever:
    def __init__(self, model_name=CE_MODEL):
        self.ce_model = CrossEncoder(model_name)

    def retrieve(self, query_text, bm25_index, dense_index,
                 chunks_by_id, top_k=TOP_K, pool_size=POOL_SIZE):
        candidates = retrieve_rrf(
            query_text, bm25_index, dense_index, top_k=pool_size
        )
        pairs = [
            (query_text, chunks_by_id[cid]["text"])
            for cid, _ in candidates
            if cid in chunks_by_id
        ]
        ce_scores = self.ce_model.predict(pairs)

        reranked = sorted(
            zip([cid for cid, _ in candidates if cid in chunks_by_id], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(score)) for cid, score in reranked[:top_k]]
