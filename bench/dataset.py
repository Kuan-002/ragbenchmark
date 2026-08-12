import json
from bench.config import MIN_QA_PER_PAPER

_DEV_JSON = "data/qasper_raw/qasper-dev-v0.3.json"


def load_qasper(json_path=_DEV_JSON):
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    papers = []
    for paper_id, paper in raw.items():
        paper = dict(paper)
        paper["id"] = paper_id
        papers.append(paper)
    print(f"Loaded {len(papers)} papers from {json_path}")
    return papers


def filter_papers(papers, min_qa=MIN_QA_PER_PAPER):
    result = []
    for paper in papers:
        # qas is a list of QA items
        n_answerable = sum(
            1 for qa in paper["qas"]
            if qa["answers"] and not qa["answers"][0]["answer"]["unanswerable"]
        )
        if n_answerable >= min_qa:
            result.append(paper)
    return result


def _get_answer_type(ans):
    if ans["yes_no"] is not None:
        return "yes_no"
    if ans["extractive_spans"]:
        return "extractive"
    if ans["free_form_answer"]:
        return "free_form"
    return "unknown"


def _answer_text(ans):
    if ans["yes_no"] is not None:
        return "yes" if ans["yes_no"] else "no"
    if ans["extractive_spans"]:
        return "; ".join(ans["extractive_spans"])
    if ans["free_form_answer"]:
        return ans["free_form_answer"]
    return ""


def extract_answerable_queries(papers):
    queries = []
    for paper in papers:
        paper_id = paper["id"]
        for qa in paper["qas"]:
            question = qa["question"]
            if not qa["answers"]:
                continue
            answer_records = [
                a["answer"] for a in qa["answers"]
                if a.get("answer") and not a["answer"]["unanswerable"]
            ]
            if not answer_records:
                continue

            evidence = []
            reference_answers = []
            for ans in answer_records:
                evidence.extend(ans.get("evidence") or [])
                answer_text = _answer_text(ans)
                if answer_text and answer_text not in reference_answers:
                    reference_answers.append(answer_text)
            if not evidence:
                continue

            text_evidence = [e for e in evidence if not e.startswith("FLOAT SELECTED")]
            if not text_evidence:
                continue
            first_answer = answer_records[0]
            queries.append({
                "qid": f"{paper_id}__{qa['question_id']}",
                "paper_id": paper_id,
                "question": question,
                "evidence": list(dict.fromkeys(text_evidence)),
                "reference_answer": " | ".join(reference_answers[:3]),
                "answer_type": _get_answer_type(first_answer),
                "nlp_background": qa.get("nlp_background"),
                "topic_background": qa.get("topic_background"),
            })
    return queries
