# Agentic RAG Benchmark Report

## Project Summary

This project upgrades a traditional paper QA RAG benchmark into a real agentic RAG system. The original retrieval pipeline mainly compared static retrieval strategies such as RRF+CE and Fusion+CE. The new implementation introduces an observation-driven tool-calling loop, where the agent can decide which tool to call, observe the result, revise its strategy, verify answer support, and stop when evidence is sufficient.

The benchmark is evaluated on QASPER-style paper QA with ground-truth evidence chunks. The key retrieval metrics are Hit@1, Hit@5, EvidenceRecall@5, MRR, NDCG@5, and AllEvidenceRecall@5. In this report, "sufficient" refers to AllEvidenceRecall@5: whether the top-5 evidence set covers all gold evidence chunks for a question.

## What Was Built

Implemented a real Agentic RAG loop with:

- Dynamic tool calling instead of a fixed retrieval workflow.
- One-tool-per-turn execution so the model must observe each result before choosing the next action.
- Dynamic tool exposure based on current state, evidence, table availability, and neighbor chunk candidates.
- Full visible trace logging for each QA: available tools, tool calls, arguments, observations, evidence before/after, verification results, and final answer.
- Claim-level answer verification before final response.

The final agent has 10 tools:

- `rewrite_query`: generate search rewrites without retrieving.
- `decompose_question`: split complex questions into subqueries.
- `retrieve_evidence`: direct factual retrieval.
- `retrieve_fusion`: multi-query fusion retrieval.
- `retrieve_numeric_or_table`: numeric/table-biased retrieval.
- `list_neighbor_chunks`: inspect adjacent chunks around a retrieved chunk.
- `extract_table_data`: extract structured table rows from loaded table evidence.
- `inspect_context`: inspect evidence sufficiency and gaps.
- `verify_answer_support`: verify candidate answer claims against current evidence.
- `finish`: stop the loop with a reason.

## Full Benchmark Result

Full evaluation used 239 papers, 845 answerable QA examples, and 10,320 filtered chunks.

| Method | Hit@1 | Hit@5 | EvidenceRecall@5 | Sufficient / AllEvidence@5 | MRR | Latency |
|---|---:|---:|---:|---:|---:|---:|
| RRF+CE | 0.1337 | 0.2911 | 0.2111 | 0.1574 | 0.1927 | 112.8 ms |
| Fusion+CE | 0.1314 | 0.2935 | 0.2122 | 0.1574 | 0.1912 | 145.2 ms |
| Agentic RAG | 0.3456 | 0.7385 | 0.5747 | 0.4521 | 0.4901 | 6366.7 ms |

## Improvement Over Traditional RAG

Compared with RRF+CE:

- Hit@1 improved by +0.2118 absolute, +158.4% relative.
- Hit@5 improved by +0.4473 absolute, +153.7% relative.
- EvidenceRecall@5 improved by +0.3636 absolute, +172.3% relative.
- Sufficient / AllEvidence@5 improved by +0.2947 absolute, +187.2% relative.
- MRR improved by +0.2974 absolute, +154.3% relative.

Compared with Fusion+CE:

- Hit@1 improved by +0.2142 absolute, +163.1% relative.
- Hit@5 improved by +0.4450 absolute, +151.6% relative.
- EvidenceRecall@5 improved by +0.3625 absolute, +170.8% relative.
- Sufficient / AllEvidence@5 improved by +0.2947 absolute, +187.2% relative.
- MRR improved by +0.2989 absolute, +156.3% relative.

## Agent Behavior Diagnostics

The full run produced 845 complete QA traces.

| Diagnostic | Value |
|---|---:|
| Average tool calls per query | 4.67 |
| Max tool calls per query | 10 |
| Multi-tool rounds | 0 |
| Verification calls | 904 |
| Verified candidate answers | 716 |
| Verification pass rate | 79.2% |
| Guardrail stops | 36 |
| Normal finish calls | 809 |

Most common tool paths:

- `retrieve_evidence -> inspect_context -> verify_answer_support -> finish`
- `retrieve_fusion -> inspect_context -> verify_answer_support -> finish`
- `retrieve_evidence -> list_neighbor_chunks -> verify_answer_support -> finish`
- `retrieve_evidence -> verify_answer_support -> finish`

This shows that the system is no longer a static retrieval workflow. Tool paths vary by question type, retrieval observation, evidence gaps, and verification result.

## Engineering Highlights

- Replaced pseudo-agentic retrieval logic with a real OpenAI-compatible tool-calling loop.
- Added stateful evidence management with deduplication, neighbor expansion, table detection, numeric evidence detection, and dynamic tool availability.
- Added trace persistence in JSONL plus a human-readable Markdown trace renderer for manual inspection.
- Added ground-truth evaluation for agentic QA with Hit@1/3/5, MRR, NDCG@5, EvidenceRecall@5, and AllEvidenceRecall@5.
- Added claim-level answer verification to reduce unsupported answers and preserve verified candidate answers.
- Added regression tests for tool loop execution, dynamic tool exposure, neighbor chunk expansion, query rewriting, and answer verification.

## Resume-Ready Bullet

Built and evaluated a real Agentic RAG system for evidence-grounded scientific paper QA on 845 QASPER examples, replacing static RRF/Fusion retrieval with an observation-driven tool-calling loop, dynamic tool exposure, neighbor chunk inspection, query rewriting, decomposition, and claim-level answer verification. Improved Hit@5 from 0.2935 to 0.7385 (+151.6%) and full-evidence sufficiency from 0.1574 to 0.4521 (+187.2%) over Fusion+CE, with complete per-question tool traces and verification logs.

## Current Limitation

The main cost is latency. Agentic RAG averages 6.37 seconds per query versus 0.11-0.15 seconds for traditional retrieval. A small number of cases also hit the max-step guardrail when verification and rewrite loops do not converge. The next optimization target is reducing unnecessary verification/rewrite cycles while keeping the evidence coverage gains.

