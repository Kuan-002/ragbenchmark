"""Local small-model router helper.

Enterprise-style routing:
1. deterministic regex handles high-precision obvious cases;
2. a small local embedding model handles softer QASPER-style phrasing;
3. if the model is missing, routing falls back to regex without crashing.

The expected model is sentence-transformers/all-MiniLM-L6-v2 saved under
models/router/all-MiniLM-L6-v2 by scripts/download_router_model.py.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


LABELS = ["numerical", "comparison", "why_how", "methodology", "factual"]
DEFAULT_MODEL_DIR = Path("models/router/all-MiniLM-L6-v2")

EXAMPLES: dict[str, list[str]] = {
    "numerical": [
        "What are the key experimental results and reported metrics?",
        "What is the performance of the model on the task?",
        "What evaluation metric do they use?",
        "What scores do they report?",
        "Which dataset achieves the best performance?",
        "How large is the dataset?",
        "How many examples are used?",
        "What are the results from the proposed strategies?",
        "What is the previous state of the art result?",
        "What BLEU score does the system achieve?",
        "What accuracy or F1 is reported?",
        "What are the baselines used in the experiments?",
        "Which benchmarks are evaluated?",
        "What languages are evaluated?",
        "What datasets did they use?",
    ],
    "comparison": [
        "How does the method compare with baselines?",
        "Which model performs better?",
        "What is the difference between the two methods?",
        "What baseline model is used?",
        "What other methods do they compare against?",
        "Does the model outperform previous work?",
        "How do their results compare to state of the art?",
        "What are the baseline systems?",
        "Which approaches are compared?",
        "What trade-off is reported between the methods?",
    ],
    "why_how": [
        "How do they obtain the dataset?",
        "How was the data collected?",
        "Why does the method improve robustness?",
        "How do they measure diversity?",
        "How are the examples annotated?",
        "How does the approach solve the problem?",
        "What is the motivation for the method?",
        "What limitation do the authors identify?",
        "How do they verify the claim?",
        "Why is the model effective?",
        "How are meaningful chains selected?",
        "How is the ground truth established?",
    ],
    "methodology": [
        "What model architecture do they use?",
        "What predictive model do they build?",
        "What is the core component of the method?",
        "How is the model implemented?",
        "What neural architecture is used?",
        "What is the training objective?",
        "What approach is proposed?",
        "What is the algorithm used?",
        "What does the model learn?",
        "How do they perform multilingual training?",
        "What is the architecture of the span detector?",
        "Which representation learning architecture do they adopt?",
        "How does their ensemble method work?",
    ],
    "factual": [
        "What is the paper title?",
        "Who are the authors?",
        "What dataset is mentioned?",
        "What task is studied?",
        "What is the name of the benchmark?",
        "Which language pair is used?",
        "What corpus is used?",
        "What is the source of the data?",
        "What does the acronym mean?",
        "Where is the method described?",
    ],
}


def model_path() -> Path:
    return Path(os.environ.get("ROUTER_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def is_model_available() -> bool:
    path = model_path()
    return path.exists() and any(path.iterdir())


@lru_cache(maxsize=1)
def _load_model():
    if not is_model_available():
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path()))


@lru_cache(maxsize=1)
def _prototype_matrix():
    model = _load_model()
    if model is None:
        return None
    texts: list[str] = []
    labels: list[str] = []
    for label in LABELS:
        for text in EXAMPLES[label]:
            texts.append(text)
            labels.append(label)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings), labels


def predict_query_type(question: str) -> dict | None:
    """Return semantic route prediction, or None if the model is unavailable."""
    model = _load_model()
    matrix = _prototype_matrix()
    if model is None or matrix is None:
        return None

    prototypes, labels = matrix
    query_vec = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = prototypes @ query_vec
    per_label = {}
    for label in LABELS:
        values = [float(score) for score, item_label in zip(sims, labels) if item_label == label]
        # Blend max and mean: max catches paraphrases, mean reduces one-example noise.
        per_label[label] = (max(values) * 0.75) + (float(np.mean(values)) * 0.25)

    ranked = sorted(per_label.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1]
    return {
        "label": best_label,
        "score": round(best_score, 4),
        "margin": round(best_score - second_score, 4),
        "scores": {label: round(score, 4) for label, score in ranked},
    }
