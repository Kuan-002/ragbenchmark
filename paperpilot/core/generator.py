"""LLM answer generation from retrieved chunks."""
import re
import os
import openai

from paperpilot.core.numeric_reasoner import numeric_reasoning_notes

_SYSTEM = """You are PaperPilot, a research assistant specializing in NLP papers.
Answer the user's question using ONLY the provided sources below.
Answer in English unless the user explicitly asks for another language.
Write a useful interview/demo answer, not a terse abstract. Cite sources as [1], [2], etc.
For conceptual or methodology questions, explain the mechanism, why it matters, and how the
parts connect. Prefer 2-4 compact paragraphs or a short structured list when that makes the
answer easier to inspect. Do not pad with generic background.
When a source is a table, read the column headers and row values carefully \
before making any numerical comparisons or calculations.
If the sources do not contain enough information, say so explicitly."""

_SOURCE_PARA   = "[{i}] (Paper: {paper}; Section: {section})\n{text}"
_SOURCE_TABLE  = "[{i}] (Paper: {paper}; Table — {caption})\n{text}"
_SOURCE_MERGED = "[{i}] (Tables for joint comparison — {caption})\n{text}"

_POLICY_TEXT = {
    "numeric_with_units": (
        "Answer with the exact number and unit/metric when present. "
        "If a calculation or comparison is needed, show the arithmetic briefly. "
        "For experimental-results questions, prioritize headline benchmark metrics "
        "and reported table values over hyperparameter or decoding settings."
    ),
    "structured_comparison": (
        "Compare both sides explicitly. Use evidence for each side; if one side "
        "is missing from the sources, say that the comparison is incomplete. "
        "Do not rank models across different papers, datasets, or tasks unless "
        "the sources report the same benchmark."
    ),
    "yes_no_then_evidence": (
        "Start with Yes, No, or Not enough information, then give the cited evidence."
    ),
    "explanatory_synthesis": (
        "Synthesize across the relevant sources instead of summarizing one passage. "
        "Explain the causal or design rationale, then connect it back to the paper's method or result."
    ),
    "method_steps": (
        "Describe the method or architecture as ordered steps/components when possible. "
        "Name each component, its role, and how it interacts with the other components."
    ),
    "concise_cited": (
        "Give a direct cited answer. If the question asks how or why, include enough detail "
        "for a reader to understand the mechanism without opening the paper."
    ),
}

_USER_TMPL = """Question: {question}

Answer policy:
{policy}

Routing checks:
{checks}

Numerical/table notes:
{numeric_notes}

Evidence quality:
{evidence_quality}

Sources:
{sources}

Answer requirements:
- Start with the direct answer in the first sentence.
- Then give the necessary explanation, mechanism, or evidence chain.
- For contribution/summary questions, identify the paper's proposed method or claim; do not describe prior work, baselines, or criticized limitations as the paper's contribution.
- Treat phrases like "previously", "traditional", "recurrent neural networks", "prior work", and "limitations" as background unless the source explicitly says the paper proposes them.
- For architecture questions, separate components and roles.
- For comparison questions, explicitly cover each side.
- For numerical questions, include exact values and units when present.
- Use citations throughout; every important factual claim should be grounded.
- If evidence is incomplete, say exactly what is missing.

Answer:"""


def _get_client() -> openai.OpenAI:
    api_key  = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?", text):
        try:
            values.append(float(raw.rstrip("%")))
        except ValueError:
            continue
    return values


def _numeric_text(chunk: dict) -> str:
    text = chunk.get("text", "")
    table_lines = [line for line in text.splitlines() if "|" in line and "---" not in line]
    return "\n".join(table_lines) if table_lines else text


def _table_numeric_notes(question: str, retrieved: list[dict], query_plan: dict | None) -> str:
    notes, facts = numeric_reasoning_notes(question, retrieved, query_plan)
    for result in retrieved:
        chunk_id = result["chunk"].get("chunk_id", "")
        result["numeric_facts"] = [
            fact for fact in facts if fact.get("chunk_id") == chunk_id
        ][:12]
    return notes


def _policy_block(query_plan: dict | None) -> tuple[str, str]:
    if not query_plan:
        return _POLICY_TEXT["concise_cited"], "- Use only the cited sources."

    style = query_plan.get("answer_style", "concise_cited")
    policy = _POLICY_TEXT.get(style, _POLICY_TEXT["concise_cited"])
    checks = [f"- {c.replace('_', ' ')}" for c in query_plan.get("confidence_checks", [])]
    entities = query_plan.get("comparison_entities") or []
    if entities:
        checks.append(f"- compare these detected entities when supported: {', '.join(entities)}")
    checks.append(f"- evidence need: {query_plan.get('evidence_need', 'single_chunk')}")
    checks.append(f"- preferred sources: {', '.join(query_plan.get('source_preference', []))}")
    return policy, "\n".join(checks)


def _evidence_quality(retrieved: list[dict]) -> str:
    if not retrieved:
        return "No sources were retrieved; answer should say there is not enough information."

    scores = [float(r.get("score", 0.0)) for r in retrieved]
    low_conf = bool(retrieved[0].get("low_confidence", False))
    papers = {
        r["chunk"].get("paper_title") or r["chunk"].get("paper_id") or ""
        for r in retrieved
    }
    papers.discard("")

    notes = [
        f"top_score={scores[0]:.2f}",
        f"all_scores_negative={all(s < 0 for s in scores)}",
        f"low_confidence={low_conf}",
    ]
    if papers:
        notes.append(f"sources_span_{len(papers)}_papers={len(papers) > 1}")
    if low_conf or all(s < 0 for s in scores):
        notes.append(
            "If evidence is weak, say the retrieved sources only partially support the answer."
        )
    if len(papers) > 1:
        notes.append(
            "If sources are from different papers, avoid treating results as a controlled head-to-head comparison."
        )
    return "\n".join(notes)


def generate_answer(question: str, retrieved: list[dict],
                    query_plan: dict | None = None) -> str:
    source_blocks: list[str] = []
    for i, r in enumerate(retrieved, 1):
        chunk = r["chunk"]
        if chunk.get("chunk_type") == "merged_tables":
            block = _SOURCE_MERGED.format(
                i=i,
                caption=chunk.get("caption", "Tables"),
                text=chunk["text"],
            )
        elif chunk.get("chunk_type") == "table":
            block = _SOURCE_TABLE.format(
                i=i,
                paper=chunk.get("paper_title", chunk.get("paper_id", "current paper")),
                caption=chunk.get("caption", chunk.get("section", "Table")),
                text=chunk["text"],
            )
        else:
            block = _SOURCE_PARA.format(
                i=i,
                paper=chunk.get("paper_title", chunk.get("paper_id", "current paper")),
                section=chunk.get("section", ""),
                text=chunk["text"],
            )
        source_blocks.append(block)

    policy, checks = _policy_block(query_plan)
    numeric_notes = _table_numeric_notes(question, retrieved, query_plan)
    evidence_quality = _evidence_quality(retrieved)

    model  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_client()
    resp   = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TMPL.format(
                question=question,
                policy=policy,
                checks=checks,
                numeric_notes=numeric_notes,
                evidence_quality=evidence_quality,
                sources="\n\n".join(source_blocks),
            )},
        ],
        temperature=0.2,
        max_tokens=850,
    )
    return resp.choices[0].message.content.strip()
