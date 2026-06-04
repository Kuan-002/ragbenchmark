import os
import random
from bench.config import LLM_PER_TYPE, BADCASE_PARTIAL_N

SUFFICIENCY_PROMPT = """You are evaluating whether retrieved paragraphs are sufficient to answer a question about a research paper.

Question: {question}

Retrieved paragraphs:
{paragraphs}

Rate the sufficiency:
- FULLY_SUFFICIENT: The paragraphs contain all information needed to answer the question completely.
- PARTIALLY_SUFFICIENT: The paragraphs contain some relevant information but are missing key details.
- INSUFFICIENT: The paragraphs do not contain useful information to answer the question.

Respond with exactly one of: FULLY_SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT
Then on a new line, briefly explain why (1-2 sentences).

Format:
VERDICT: <verdict>
REASON: <reason>"""

BADCASE_PROMPT = """You are diagnosing why a retrieval system failed to return sufficient context for a question.

Question: {question}
Gold evidence (what should have been retrieved): {gold_text}
Retrieved paragraphs (what was actually returned): {retrieved_text}

Classify the failure into exactly one category:
- RETRIEVAL_FAILURE: The correct paragraph exists in the corpus but was not retrieved or ranked low. (Fixable with better retrieval)
- MISSING_EVIDENCE: The answer cannot be found in any paragraph — it may be in a table, figure, or simply not present in the text. (Not fixable without adding data)
- REDUNDANCY: Multiple retrieved paragraphs repeat the same information, crowding out diverse evidence. (Fixable with deduplication)

Respond with:
CATEGORY: <category>
FIXABLE: YES or NO
REASON: <1-2 sentences>"""


def _call_llm(prompt, llm_client):
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content


def judge_sufficiency(question, retrieved_ids, chunks_by_id, llm_client):
    paragraphs = "\n\n".join([
        f"[Paragraph {i+1}]\n{chunks_by_id[cid]['text']}"
        for i, cid in enumerate(retrieved_ids)
        if cid in chunks_by_id
    ])
    text = _call_llm(
        SUFFICIENCY_PROMPT.format(question=question, paragraphs=paragraphs),
        llm_client,
    )
    for label in ["FULLY_SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT"]:
        if label in text:
            return label
    return "INSUFFICIENT"


def judge_badcase_category(question, gold_ids, retrieved_ids, chunks_by_id, llm_client):
    gold_text = "\n\n".join([
        chunks_by_id[cid]["text"] for cid in gold_ids if cid in chunks_by_id
    ]) or "(not found in corpus)"
    retrieved_text = "\n\n".join([
        f"[{i+1}] {chunks_by_id[cid]['text']}"
        for i, cid in enumerate(retrieved_ids)
        if cid in chunks_by_id
    ])
    text = _call_llm(
        BADCASE_PROMPT.format(
            question=question,
            gold_text=gold_text,
            retrieved_text=retrieved_text,
        ),
        llm_client,
    )
    for label in ["RETRIEVAL_FAILURE", "MISSING_EVIDENCE", "REDUNDANCY"]:
        if label in text:
            return label
    return "RETRIEVAL_FAILURE"


def is_fixable(category):
    return category in ("RETRIEVAL_FAILURE", "REDUNDANCY")


def stratified_sample(query_results, n_per_type=LLM_PER_TYPE, seed=42):
    random.seed(seed)
    by_type = {}
    for r in query_results:
        t = r.query_type
        by_type.setdefault(t, []).append(r)

    sampled = []
    for t, items in by_type.items():
        k = min(n_per_type, len(items))
        sampled.extend(random.sample(items, k))
    return sampled


def run_two_pass_judge(query_results, sampled_results, chunks_by_id,
                       llm_client, badcase_partial_n=BADCASE_PARTIAL_N):
    sampled_qids = {r.qid for r in sampled_results}

    # Pass 1: sufficiency
    sufficiency_map = {}
    for r in sampled_results:
        print(f"  [Pass1] Judging {r.qid[:40]}...")
        verdict = judge_sufficiency(r.question, r.retrieved_ids, chunks_by_id, llm_client)
        sufficiency_map[r.qid] = verdict
        r.sufficiency = verdict

    # Pass 2: badcase root cause
    insufficient = [r for r in sampled_results if sufficiency_map.get(r.qid) == "INSUFFICIENT"]
    partial = sorted(
        [r for r in sampled_results if sufficiency_map.get(r.qid) == "PARTIALLY_SUFFICIENT"],
        key=lambda x: x.ndcg5,
    )[:badcase_partial_n]

    badcase_pool = insufficient + partial
    for r in badcase_pool:
        print(f"  [Pass2] Badcase analysis {r.qid[:40]}...")
        category = judge_badcase_category(
            r.question, r.gold_ids, r.retrieved_ids, chunks_by_id, llm_client
        )
        r.badcase_category = category

    return sufficiency_map, badcase_pool
