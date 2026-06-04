DENSE_MODEL  = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
CE_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K             = 60
CANDIDATE_K       = 100
POOL_SIZE         = 50
BADCASE_PARTIAL_N = 20
LLM_PER_TYPE      = 10
JACCARD_THRESHOLD = 0.4
MIN_QA_PER_PAPER  = 2
TOP_K             = 5

EMBEDDING_CACHE = "data/embeddings_mpnet.npy"
RESULTS_CACHE   = "data/results_latest.json"
