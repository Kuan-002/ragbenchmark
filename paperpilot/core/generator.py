"""LLM answer generation from retrieved chunks."""
import os
import openai

_SYSTEM = """You are PaperPilot, a research assistant specializing in NLP papers.
Answer the user's question using ONLY the provided sources below.
Be concise (2-4 sentences). Cite sources as [1], [2], etc.
When a source is a table, read the column headers and row values carefully \
before making any numerical comparisons or calculations.
If the sources do not contain enough information, say so explicitly."""

_SOURCE_PARA   = "[{i}] (Section: {section})\n{text}"
_SOURCE_TABLE  = "[{i}] (Table — {caption})\n{text}"
_SOURCE_MERGED = "[{i}] (Tables for joint comparison — {caption})\n{text}"

_USER_TMPL = """Question: {question}

Sources:
{sources}

Answer:"""


def _get_client() -> openai.OpenAI:
    api_key  = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def generate_answer(question: str, retrieved: list[dict]) -> str:
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
                caption=chunk.get("caption", chunk.get("section", "Table")),
                text=chunk["text"],
            )
        else:
            block = _SOURCE_PARA.format(
                i=i,
                section=chunk.get("section", ""),
                text=chunk["text"],
            )
        source_blocks.append(block)

    model  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_client()
    resp   = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TMPL.format(
                question=question,
                sources="\n\n".join(source_blocks),
            )},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()
