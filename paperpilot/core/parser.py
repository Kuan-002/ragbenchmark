"""PDF and arXiv paper parser → paragraph + table chunks via GROBID.

GROBID is the same pipeline used by S2ORC / QASPER, so paragraph boundaries
are structurally comparable to benchmark ground truth.  Tables are serialised
from the TEI <figure type="table"> elements and indexed alongside paragraphs.

Requires a running GROBID instance (default: http://localhost:8070).
Quick start:
    docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1
"""
import re
import os
import xml.etree.ElementTree as ET
import httpx

_GROBID_URL        = os.environ.get("GROBID_URL", "http://localhost:8070")
_MAX_CHUNK_CHARS   = 1600
_MERGE_CHUNK_CHARS = 200   # paragraphs shorter than this are merged with neighbours
_TEI_NS            = "http://www.tei-c.org/ns/1.0"


def _tei(tag: str) -> str:
    return f"{{{_TEI_NS}}}{tag}"


def _clean(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


# ── Table serialisation ───────────────────────────────────────────────────────

def _extract_table_text(fig_el) -> str | None:
    """Serialise a GROBID <figure type="table"> element to Markdown.

    Format:
        {caption}
        | col1 | col2 | ... |
        |---|---|---|
        | val1 | val2 | ... |

    Header rows are detected via role="head" on <cell> elements.
    When no role attribute is present, the first row is treated as the header.
    Returns None when the table has no extractable rows.
    """
    head = fig_el.find(_tei("head"))
    if head is None:
        head = fig_el.find(_tei("figDesc"))
    caption = _clean(head) if head is not None else ""

    table_el = fig_el.find(_tei("table"))
    if table_el is None:
        return None

    md_rows:    list[str] = []
    sep_added              = False
    prev_was_header        = False

    for row_idx, row in enumerate(table_el.findall(_tei("row"))):
        cells     = row.findall(_tei("cell"))
        cell_vals = [_clean(c) for c in cells]
        if not any(cell_vals):
            continue

        is_header = (
            any(c.get("role") == "head" for c in cells)
            or (row_idx == 0 and not sep_added)   # first row as fallback
        )

        md_row = "| " + " | ".join(v or " " for v in cell_vals) + " |"
        md_rows.append(md_row)

        if is_header and not sep_added:
            n_cols = max(len(cell_vals), 1)
            md_rows.append("|" + "---|" * n_cols)
            sep_added = True
        prev_was_header = is_header

    if not md_rows:
        return None

    parts = ([caption] if caption else []) + md_rows
    return "\n".join(parts)


def _table_index_text(fig_el, caption: str) -> str:
    """Flat token string for BM25 indexing: caption + all unique cell values.

    Structural chars (|, -, newlines) are stripped so numbers tokenise cleanly.
    e.g. "28.4" becomes tokens ["28", "4"] without pipe noise.
    """
    table_el = fig_el.find(_tei("table"))
    if table_el is None:
        return caption

    words = [caption] if caption else []
    seen: set[str] = set()
    for row in table_el.findall(_tei("row")):
        for cell in row.findall(_tei("cell")):
            val = _clean(cell)
            if val and val not in seen:
                words.append(val)
                seen.add(val)
    return " ".join(words)


# ── Body extraction ───────────────────────────────────────────────────────────

def _extract_divs(parent, current_section: str, chunks: list[dict]) -> None:
    """Recursively walk <div> elements, collecting <p> paragraphs and tables."""
    for div in parent.findall(_tei("div")):
        head    = div.find(_tei("head"))
        section = _clean(head) if head is not None else current_section

        for p in div.findall(_tei("p")):
            text = _clean(p)
            if text:
                chunks.append({
                    "section":    section,
                    "text":       text,
                    "chunk_type": "paragraph",
                })

        for fig in div.findall(_tei("figure")):
            if fig.get("type") == "table":
                text = _extract_table_text(fig)
                if text:
                    head_el = fig.find(_tei("head"))
                    caption = _clean(head_el) if head_el is not None else "Table"
                    chunks.append({
                        "section":    section,
                        "text":       text,
                        "index_text": _table_index_text(fig, caption),
                        "chunk_type": "table",
                        "caption":    caption,
                    })

        _extract_divs(div, section, chunks)


# ── Short-chunk merging ───────────────────────────────────────────────────────

def _merge_short_chunks(chunks: list[dict]) -> list[dict]:
    """Merge consecutive short paragraphs within the same section.

    Paragraphs shorter than _MERGE_CHUNK_CHARS are accumulated into a buffer
    and flushed when the buffer grows past the threshold, the section changes,
    or a table chunk is encountered.  Tables are never merged: even a small
    table carries structured numerical data and must stay as its own chunk.
    """
    result:  list[dict]  = []
    pending: dict | None = None

    def flush():
        if pending is not None:
            result.append(pending)

    for chunk in chunks:
        if chunk["chunk_type"] != "paragraph":
            flush()
            pending = None
            result.append(chunk)
            continue

        if pending is None:
            if len(chunk["text"]) < _MERGE_CHUNK_CHARS:
                pending = chunk
            else:
                result.append(chunk)
        else:
            if chunk["section"] == pending["section"]:
                merged_text = pending["text"] + " " + chunk["text"]
                pending = {**pending, "text": merged_text}
                if len(pending["text"]) >= _MERGE_CHUNK_CHARS:
                    result.append(pending)
                    pending = None
            else:
                # Section boundary: flush current buffer, start fresh
                flush()
                pending = chunk if len(chunk["text"]) < _MERGE_CHUNK_CHARS else None
                if pending is None:
                    result.append(chunk)

    flush()
    return result


# ── Safety split ──────────────────────────────────────────────────────────────

def _split_long_chunk(chunk: dict) -> list[dict]:
    text = chunk["text"]
    if len(text) <= _MAX_CHUNK_CHARS:
        return [chunk]

    sentences   = re.split(r"(?<=[.!?])\s+", text)
    sub_chunks: list[str] = []
    current: list[str]    = []
    current_len           = 0

    for sent in sentences:
        if current_len + len(sent) > _MAX_CHUNK_CHARS and current:
            sub_chunks.append(" ".join(current))
            current     = [current[-1], sent]
            current_len = len(current[0]) + 1 + len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        sub_chunks.append(" ".join(current))

    return [{**chunk, "text": sub} for sub in sub_chunks if sub]


# ── GROBID call ───────────────────────────────────────────────────────────────

def _call_grobid(file_bytes: bytes) -> bytes:
    try:
        resp = httpx.post(
            f"{_GROBID_URL}/api/processFulltextDocument",
            files={"input": ("paper.pdf", file_bytes, "application/pdf")},
            data={"consolidateHeader": "0"},
            timeout=120,
        )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach GROBID at {_GROBID_URL}. "
            "Start with: docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1"
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"GROBID returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    return resp.content


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> tuple[dict, list[dict]]:
    """Parse PDF via GROBID → (metadata, chunks).

    Each chunk: {chunk_id, section, text, chunk_type}
    chunk_type is "paragraph" or "table".
    """
    root = ET.fromstring(_call_grobid(file_bytes))

    title_el = root.find(f".//{_tei('title')}[@type='main']")
    title    = _clean(title_el) if title_el is not None else ""

    abstract = ""
    abs_el   = root.find(f".//{_tei('abstract')}")
    if abs_el is not None:
        parts    = [_clean(p) for p in abs_el.findall(f".//{_tei('p')}")]
        abstract = re.sub(r"\s+", " ", " ".join(parts)).strip()[:1000]

    raw: list[dict] = []
    body = root.find(f".//{_tei('body')}")
    if body is not None:
        _extract_divs(body, "Unknown", raw)

    merged = _merge_short_chunks(raw)
    chunks = [sub for c in merged for sub in _split_long_chunk(c)]

    for i, c in enumerate(chunks):
        c["chunk_id"] = f"chunk_{i:04d}"

    return {"title": title, "abstract": abstract}, chunks


def parse_arxiv(arxiv_id: str) -> tuple[dict, list[dict]]:
    arxiv_id = arxiv_id.strip().replace("https://arxiv.org/abs/", "")
    pdf_url  = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(pdf_url)
        if resp.status_code != 200:
            raise ValueError(
                f"Could not download arXiv paper {arxiv_id} "
                f"(HTTP {resp.status_code})"
            )
    return parse_pdf(resp.content)
