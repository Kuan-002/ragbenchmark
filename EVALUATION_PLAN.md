# Evaluation Plan

## Dataset Fit

This project uses QASPER dev as an evidence-grounded RAG evaluation set. The
dataset includes full paper text, questions, answer types, reference answers,
and annotated evidence. In the current aligned split, most questions have one
gold evidence chunk, so ranking metrics designed for many graded relevant
documents are not the best primary signal.

## Retriever Metrics

Primary:

- Hit@1, Hit@3, Hit@5
- MRR
- First Gold Rank
- Evidence Recall@5
- All Evidence Recall@5

Secondary:

- NDCG@5
- Section redundancy
- Latency

NDCG@5 is retained to compare with older reports, but the project treats it as
a secondary ranking diagnostic.

## Context Metrics

Use stratified, reference-aware LLM judging by query type. The judge receives
the question, the reference answer, and the retrieved top-k context. It scores
only whether the context contains the information needed to answer the question,
not whether a generated answer is well written.

- FULLY_SUFFICIENT
- PARTIALLY_SUFFICIENT
- INSUFFICIENT

The structured judge also returns:

- score from 1 to 5
- missing facts
- one-sentence rationale

This layer answers whether the retrieved top-k context is enough for generation,
which pure retrieval metrics cannot determine.

Production use should calibrate this judge against a small human-labeled set and
track judge agreement before treating the metric as authoritative.

## Answer Metrics

Planned answer-level metrics:

- Correctness against the reference answer
- Faithfulness to retrieved context
- Citation accuracy
- Yes/no accuracy for binary questions
- Extractive-span containment or token F1 for extractive questions

Free-form answers should be evaluated with a deterministic LLM judge and a small
human-labeled calibration set.

## Breakdown Axes

Report metrics by:

- Query type: factual, methodology, why/how, numerical, comparison
- Answer type: extractive, free_form, yes_no
- Evidence count: single evidence vs multi evidence
- User background metadata when available

## Known Limits

- Floating figure evidence is excluded from the text-only retrieval benchmark.
- Table evidence should be evaluated separately once table extraction coverage
  is stable.
- Multiple annotators should be merged in a future version instead of only using
  the first answer annotation.
