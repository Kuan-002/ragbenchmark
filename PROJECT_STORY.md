# PaperPilot Project Story

## One-Line Positioning

PaperPilot is an evidence-grounded RAG benchmark that evolved into a
source-cited knowledge assistant for research and technical documents.

## Problem

Long technical documents are hard to query reliably. A naive RAG system often
fails before generation: the answer exists, but the right evidence is not in
the retrieved context, or the retrieved context contains only one side of a
comparison, misses a table, or lacks enough paragraphs for synthesis.

## Benchmark First

The first version evaluated retrieval strategies on QASPER before choosing a
product pipeline. It compared BM25, dense retrieval, RRF, and CrossEncoder
reranking. The benchmark originally reported NDCG@5, Recall@5, and MRR.

The current evaluation reframes QASPER as an evidence-grounded RAG dataset:

- Retriever diagnostics: Hit@1/3/5, MRR, Gold Rank, Evidence Recall@5
- Context diagnostics: whether top-k evidence is sufficient to answer
- Answer diagnostics: correctness, faithfulness, citation accuracy

NDCG@5 remains a secondary ranking signal, not the main success metric.

## Product Translation

The benchmark findings were converted into PaperPilot Web:

- PDF and arXiv ingestion through GROBID
- Paragraph and table chunks
- Table-aware BM25 indexing
- QueryPlan routing for factual, methodology, why/how, numerical, and comparison questions
- BM25 plus CrossEncoder retrieval
- Cited answer generation
- User feedback logging for retrieval and answer failure analysis

## Retrieval Upgrade

The project uses explicit retrieval policies instead of an agent loop:

1. Classify query intent
2. Choose the retrieval strategy
3. Retrieve paragraph and table evidence
4. Generate a grounded cited answer

## Why This Design

Technical answers need traceability and stable evidence more than open-ended
autonomy. Explicit retrieval policies keep latency and failure modes controlled
while still supporting different evidence needs for factual, numerical,
comparison, methodology, and why/how questions.

## Current Limitations

- No visual understanding for figures and screenshots
- GROBID quality limits table extraction
- LLM judges need calibration against a small human-labeled set
- Feedback is logged locally and not yet aggregated into a dashboard
- The demo is optimized for NLP papers, not all technical domains

## Next Steps

- Add answer-level evaluation for correctness, faithfulness, and citation accuracy
- Add feedback analytics for badcase trends by query type and retrieval mode
- Extend the corpus from papers to API docs, SDK examples, and troubleshooting guides
- Add version-aware document indexing for technical content synchronization
