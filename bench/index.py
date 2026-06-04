import re
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from bench.config import DENSE_MODEL, EMBEDDING_CACHE


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def get_scores(self, query_text):
        tokens = tokenize(query_text)
        return self.bm25.get_scores(tokens)


class DenseIndex:
    def __init__(self, chunks, model_name=DENSE_MODEL, cache_path=EMBEDDING_CACHE):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.model = SentenceTransformer(model_name)

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            print(f"Loaded embeddings from cache: {cache_path}")
        else:
            texts = [c["text"] for c in chunks]
            self.embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self.embeddings)
            print(f"Saved embeddings to: {cache_path}")

    def encode_query(self, query_text):
        return self.model.encode([query_text], normalize_embeddings=True)[0]

    def get_scores(self, query_text):
        q_emb = self.encode_query(query_text)
        return self.embeddings @ q_emb
