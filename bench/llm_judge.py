import json
import os
import random
import re

from bench.config import BADCASE_PARTIAL_N, LLM_PER_TYPE

SUFFICIENCY_PROMPT = """You are an evaluator for evidence-grounded RAG.

Task:
Judge whether the retrieved context contains enough information to answer the
question, using the reference answer as the target information need. Do not
judge writing quality. Judge only whether the context contains the needed facts.

Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{paragraphs}

Rubric:
5 = fully sufficient: context directly contains all critical facts needed.
4 = mostly sufficient: context contains the answer, with only minor missing details.
3 = partially sufficient: context contains useful related evidence but misses important facts.
2 = weakly sufficient: context is only tangentially related.
1 = insufficient: context does not contain useful evidence for the answer.

Return JSON only:
{{
  "score": 1,
  "verdict": "FULLY_SUFFICIENT|PARTIALLY_SUFFICIENT|INSUFFICIENT",
  "missing_facts": ["short missing fact"],
  "reason": "one sentence"
}}

Verdict mapping:
- score 4-5: FULLY_SUFFICIENT
- score 3: PARTIALLY_SUFFICIENT
- score 1-2: INSUFFICIENT
"""

BADCASE_PROMPT = """You are diagnosing why a retrieval system failed to return sufficient context for a question.

Question: {question}
Reference answer: {reference_answer}
Gold evidence: {gold_text}
Retrieved paragraphs: {retrieved_text}

Classify the failure into exactly one category:
- RETRIEVAL_FAILURE: The correct evidence exists in the corpus but was not retrieved or ranked high enough.
- MISSING_EVIDENCE: The answer cannot be found in text chunks; it may require a table, figure, or absent source.
- REDUNDANCY: Retrieved chunks repeat the same point and crowd out diverse required evidence.

Return JSON only:
{{
  "category": "RETRIEVAL_FAILURE|MISSING_EVIDENCE|REDUNDANCY",
  "fixable": true,
  "reason": "one sentence"
}}"""


def _call_llm(prompt, llm_client, max_tokens=300):
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response did not contain JSON")
    return json.loads(text[start:end + 1])


def _verdict_from_score(score):
    if score >= 4:
        return "FULLY_SUFFICIENT"
    if score == 3:
        return "PARTIALLY_SUFFICIENT"
    return "INSUFFICIENT"


def _context_text(retrieved_ids, chunks_by_id):
    return "\n\n".join([
        f"[Chunk {i + 1} | {cid}]\n{chunks_by_id[cid]['text']}"
        for i, cid in enumerate(retrieved_ids)
        if cid in chunks_by_id
    ])


def judge_sufficiency_detail(result, chunks_by_id, llm_client):
    paragraphs = _context_text(result.retrieved_ids, chunks_by_id)
    reference_answer = result.reference_answer or "(no reference answer text)"
    try:
        raw = _call_llm(
            SUFFICIENCY_PROMPT.format(
                question=result.question,
                reference_answer=reference_answer,
                paragraphs=paragraphs,
            ),
            llm_client,
        )
        data = _json_object(raw)
    except Exception as exc:
        data = {
            "score": 1,
            "verdict": "INSUFFICIENT",
            "missing_facts": ["judge_parse_error"],
            "reason": f"Judge response could not be parsed: {exc}",
        }

    try:
        score = int(data.get("score", 1))
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(5, score))
    verdict = data.get("verdict")
    if verdict not in {"FULLY_SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT"}:
        verdict = _verdict_from_score(score)

    missing = data.get("missing_facts") or []
    if not isinstance(missing, list):
        missing = [str(missing)]

    return {
        "score": score,
        "verdict": verdict,
        "missing_facts": [str(item)[:160] for item in missing[:5]],
        "reason": str(data.get("reason", ""))[:500],
    }


def judge_sufficiency(question, retrieved_ids, chunks_by_id, llm_client,
                      reference_answer=None):
    class _Result:
        pass

    result = _Result()
    result.question = question
    result.reference_answer = reference_answer
    result.retrieved_ids = retrieved_ids
    return judge_sufficiency_detail(result, chunks_by_id, llm_client)["verdict"]


def judge_badcase_category(result, chunks_by_id, llm_client):
    gold_text = "\n\n".join([
        chunks_by_id[cid]["text"] for cid in result.gold_ids if cid in chunks_by_id
    ]) or "(not found in corpus)"
    retrieved_text = _context_text(result.retrieved_ids, chunks_by_id)
    try:
        raw = _call_llm(
            BADCASE_PROMPT.format(
                question=result.question,
                reference_answer=result.reference_answer or "(no reference answer text)",
                gold_text=gold_text,
                retrieved_text=retrieved_text,
            ),
            llm_client,
        )
        data = _json_object(raw)
        category = data.get("category", "RETRIEVAL_FAILURE")
    except Exception:
        category = "RETRIEVAL_FAILURE"

    if category not in {"RETRIEVAL_FAILURE", "MISSING_EVIDENCE", "REDUNDANCY"}:
        category = "RETRIEVAL_FAILURE"
    return category


def is_fixable(category):
    return category in ("RETRIEVAL_FAILURE", "REDUNDANCY")


def stratified_sample(query_results, n_per_type=LLM_PER_TYPE, seed=42):
    random.seed(seed)
    by_type = {}
    for r in query_results:
        by_type.setdefault(r.query_type, []).append(r)

    sampled = []
    for _, items in by_type.items():
        k = min(n_per_type, len(items))
        sampled.extend(random.sample(items, k))
    return sampled


def run_two_pass_judge(query_results, sampled_results, chunks_by_id,
                       llm_client, badcase_partial_n=BADCASE_PARTIAL_N):
    sufficiency_map = {}
    for r in sampled_results:
        print(f"  [Pass1] Judging {r.qid[:40]}...")
        detail = judge_sufficiency_detail(r, chunks_by_id, llm_client)
        sufficiency_map[r.qid] = detail
        r.sufficiency = detail["verdict"]
        r.sufficiency_score = detail["score"]
        r.sufficiency_reason = detail["reason"]
        r.sufficiency_missing = detail["missing_facts"]

    insufficient = [
        r for r in sampled_results
        if sufficiency_map.get(r.qid, {}).get("verdict") == "INSUFFICIENT"
    ]
    partial = sorted(
        [
            r for r in sampled_results
            if sufficiency_map.get(r.qid, {}).get("verdict") == "PARTIALLY_SUFFICIENT"
        ],
        key=lambda x: (x.sufficiency_score or 0, x.hit5, x.mrr),
    )[:badcase_partial_n]

    badcase_pool = insufficient + partial
    for r in badcase_pool:
        print(f"  [Pass2] Badcase analysis {r.qid[:40]}...")
        r.badcase_category = judge_badcase_category(r, chunks_by_id, llm_client)

    return sufficiency_map, badcase_pool
