"""PaperPilot FastAPI backend."""
import os, json, uuid, logging
from pathlib import Path

_DEMO_DIR = Path("demo_papers")
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

# Load .env
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from paperpilot.core.indexer       import load_models, PaperIndex
from paperpilot.core.library_index import LibraryIndex
from paperpilot.core.parser        import parse_pdf
from paperpilot.core.classifier    import get_hint
from paperpilot.core.generator     import generate_answer
from paperpilot.core.router        import route_query
from paperpilot.core.agentic_rag   import run_agentic_rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paperpilot")

_LIBRARY_INDEX_DIR = Path(os.environ.get("LIBRARY_INDEX_DIR", "library_index"))

# ── Global state ───────────────────────────────────────────────────────────────
sessions:      dict[str, dict] = {}   # Upload Mode sessions
library_index: LibraryIndex | None = None

FEEDBACK_LOG = Path("data/feedback.jsonl")
FEEDBACK_LOG.parent.mkdir(exist_ok=True)


def _get_llm_client():
    import openai
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url and os.environ.get("DEEPSEEK_API_KEY"):
        base_url = "https://api.deepseek.com"
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url or None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global library_index
    log.info("Loading CrossEncoder …")
    load_models()
    log.info("Models ready.")

    # Try to load Library Mode index (optional — upload mode works without it)
    chunks_file = _LIBRARY_INDEX_DIR / "library_chunks.json"
    if chunks_file.exists():
        try:
            log.info(f"Loading library index from {_LIBRARY_INDEX_DIR} …")
            library_index = LibraryIndex(_LIBRARY_INDEX_DIR)
            log.info(f"Library ready: {library_index.n_papers} papers, "
                     f"{len(library_index.chunks)} chunks.")
        except Exception as e:
            log.warning(f"Library index load failed: {e}. Library Mode unavailable.")
    else:
        log.info("No library index found. Library Mode unavailable. "
                 "Run scripts/build_library_index.py to enable it.")
    yield


app = FastAPI(title="PaperPilot", lifespan=lifespan)


@app.middleware("http")
async def no_store_ui_assets(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/library"} or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── Static files ───────────────────────────────────────────────────────────────
_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
def root(request: Request):
    if not request.url.query:
        return RedirectResponse("/?ui=agentic-demo-9", status_code=307)
    return FileResponse(str(_static / "index.html"))


@app.get("/library")
def library_page(request: Request):
    if not request.url.query:
        return RedirectResponse("/library?ui=en-only-2", status_code=307)
    return FileResponse(str(_static / "library.html"))


# ── Upload Mode endpoints ──────────────────────────────────────────────────────

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    try:
        metadata, chunks = parse_pdf(content)
    except Exception as e:
        raise HTTPException(422, f"PDF parse error: {e}")

    session_id = str(uuid.uuid4())
    log.info(f"Building index for session {session_id} ({len(chunks)} chunks)…")
    _stamp_session_chunks(chunks, metadata, session_id)
    index = PaperIndex(chunks)
    sessions[session_id] = {"metadata": metadata, "index": index}
    log.info(f"Index ready: {session_id}")

    return {
        "session_id": session_id,
        "title":      metadata["title"],
        "abstract":   metadata["abstract"],
        "num_chunks": len(chunks),
        "sections":   list(dict.fromkeys(c["section"] for c in chunks)),
    }


class QueryRequest(BaseModel):
    session_id: str
    question:   str
    mode:       str = "agentic"
    top_k:      int = 5


@app.post("/api/query")
async def query(req: QueryRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload a paper first.")

    index: PaperIndex = session["index"]
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")

    llm_client  = _get_llm_client()
    query_plan  = route_query(question, llm_client=llm_client)
    query_type  = query_plan.query_type
    hint        = get_hint(query_type)

    mode = req.mode.lower().strip()
    if mode in {"agentic", "agentic_rag", "agentic-rag"}:
        if llm_client is None:
            raise HTTPException(
                400,
                "Agentic-RAG requires OPENAI_API_KEY or DEEPSEEK_API_KEY.",
            )

        plan_dict = query_plan.to_dict()

        def retrieve_fn(search_query: str, top_k: int) -> list[dict]:
            search_type = plan_dict.get("query_type", query_type)
            if search_type in {"comparison", "why_how", "methodology"}:
                return index.retrieve_multi_query_ce(search_query, plan_dict, top_k=top_k)
            return index.retrieve_ce(
                search_query,
                top_k=top_k,
                cross_table=plan_dict.get("cross_table", False),
                query_type=search_type,
            )

        neighbor_fn = _make_neighbor_fn(index)
        agent_result = run_agentic_rag(
            question,
            plan_dict,
            retrieve_fn,
            llm_client,
            top_k=req.top_k,
            neighbor_fn=neighbor_fn,
            max_steps=7,
        )
        retrieved = agent_result["retrieved"]
        answer = agent_result["answer"]
        response_mode = "agentic"
        method_label = "Agentic-RAG"
        agent_trace = _fmt_agent_trace(agent_result.get("rounds", []))
    elif mode in {"rrf_ce", "traditional", "ce", "rrf"}:
        # Single-paper demo path uses the project's established BM25 candidate
        # retrieval plus CrossEncoder reranking path.
        retrieved = index.retrieve_ce(question, top_k=req.top_k,
                                      cross_table=query_plan.cross_table,
                                      query_type=query_type)
        answer = generate_answer(question, retrieved, query_plan.to_dict())
        response_mode = "rrf_ce"
        method_label = "Traditional BM25+CE"
        agent_trace = []
    else:
        raise HTTPException(400, f"Unsupported mode: {req.mode}")

    low_confidence = bool(retrieved[0].get("low_confidence", False)) if retrieved else True
    expected_title = session.get("metadata", {}).get("title", "")
    mismatched = [
        r["chunk"].get("chunk_id", "")
        for r in retrieved
        if expected_title and r["chunk"].get("paper_title") not in {"", expected_title}
    ]
    if mismatched:
        log.warning(
            "Retrieved chunks from unexpected paper for session %s: %s",
            req.session_id,
            mismatched,
        )

    return {
        "question":       question,
        "paper_title":    session.get("metadata", {}).get("title", ""),
        "query_type":     query_type,
        "query_plan":     query_plan.to_dict(),
        "hint":           hint,
        "answer":         answer,
        "sources":        _fmt_sources(retrieved),
        "mode":           response_mode,
        "method_label":   method_label,
        "agent_trace":    agent_trace,
        "low_confidence": low_confidence,
    }


# ── Demo papers ───────────────────────────────────────────────────────────────

@app.get("/api/demo/papers")
def list_demo_papers():
    if not _DEMO_DIR.exists():
        return {"papers": []}
    papers = []
    for path in sorted(_DEMO_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            papers.append({
                "key":          data.get("key", path.stem),
                "display_name": data.get("display_name", path.stem),
                "title":        data.get("metadata", {}).get("title", ""),
                "num_chunks":   len(data.get("chunks", [])),
            })
        except Exception:
            continue
    return {"papers": papers}


@app.post("/api/demo/load")
async def load_demo_paper(paper_key: str = Form(...)):
    path = _DEMO_DIR / f"{paper_key}.json"
    if not path.exists():
        raise HTTPException(404, f"Demo paper '{paper_key}' not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read demo paper: {e}")

    metadata = dict(data.get("metadata", {}))
    if data.get("display_name"):
        metadata["title"] = data["display_name"]
    chunks   = data.get("chunks", [])
    if not chunks:
        raise HTTPException(422, "Demo paper has no chunks")

    session_id = str(uuid.uuid4())
    log.info(f"Loading demo '{paper_key}' into session {session_id} ({len(chunks)} chunks)")
    _stamp_session_chunks(chunks, metadata, session_id, paper_key=paper_key)
    index = PaperIndex(chunks)
    sessions[session_id] = {"metadata": metadata, "index": index}

    return {
        "session_id": session_id,
        "title":      metadata.get("title", ""),
        "abstract":   metadata.get("abstract", ""),
        "num_chunks": len(chunks),
        "sections":   list(dict.fromkeys(c["section"] for c in chunks)),
    }


# ── Feedback ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id:   str
    question:     str
    helpful:      bool
    failure_type: str | None = None
    mode:         str | None = None
    query_type:   str | None = None
    answer:       str | None = None
    retrieved_chunk_ids: list[str] = []


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    record = {
        "ts":           datetime.utcnow().isoformat(),
        "session_id":   req.session_id,
        "question":     req.question,
        "helpful":      req.helpful,
        "failure_type": req.failure_type,
        "mode":         req.mode,
        "query_type":   req.query_type,
        "answer":       req.answer,
        "retrieved_chunk_ids": req.retrieved_chunk_ids,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


# ── Library Mode endpoints ─────────────────────────────────────────────────────

def _require_library():
    if library_index is None:
        raise HTTPException(
            503,
            "Library index not loaded. "
            "Run scripts/build_library_index.py and restart the server."
        )
    return library_index


@app.get("/api/library/search")
async def library_search(
    query:  str = Query(..., description="Search question"),
    top_k:  int = Query(5,   description="Number of results"),
):
    lib = _require_library()
    if not query.strip():
        raise HTTPException(400, "Empty query")

    llm_client = _get_llm_client()
    query_plan = route_query(query, llm_client=llm_client)
    query_type = query_plan.query_type
    hint       = get_hint(query_type)

    result     = lib.search(query, top_k=top_k, llm_client=llm_client,
                            query_type=query_type,
                            cross_table=query_plan.cross_table)
    retrieved  = result["results"]
    answer = generate_answer(query, retrieved, query_plan.to_dict())

    low_confidence = bool(retrieved[0].get("low_confidence", False)) if retrieved else True

    sources = []
    for i, r in enumerate(_fmt_sources(retrieved), 1):
        r["paper_id"]    = retrieved[i - 1]["chunk"].get("paper_id", "")
        r["paper_title"] = retrieved[i - 1]["chunk"].get("paper_title", "")
        sources.append(r)

    return {
        "query":          query,
        "query_type":     query_type,
        "query_plan":     query_plan.to_dict(),
        "hint":           hint,
        "strategy":       result["strategy"],
        "mode":           "ce",
        "answer":         answer,
        "results":        sources,
        "low_confidence": low_confidence,
        "latency_ms":     round(result["latency_ms"], 1),
    }


@app.get("/api/library/papers")
async def library_papers():
    lib = _require_library()
    return {"papers": lib.paper_list()}


@app.get("/api/library/stats")
async def library_stats():
    lib = _require_library()
    return lib.stats()


# ── Shared helper ──────────────────────────────────────────────────────────────

def _stamp_session_chunks(chunks: list[dict], metadata: dict,
                          session_id: str, paper_key: str | None = None) -> None:
    title = (
        metadata.get("title")
        or metadata.get("paper_title")
        or paper_key
        or "current paper"
    )
    paper_id = paper_key or metadata.get("paper_id") or session_id
    for chunk in chunks:
        original_id = chunk.get("chunk_id", "")
        chunk.setdefault("original_chunk_id", original_id)
        chunk["paper_title"] = title
        chunk["paper_id"] = paper_id
        if original_id and not str(original_id).startswith(f"{paper_id}:"):
            chunk["chunk_id"] = f"{paper_id}:{original_id}"


def _make_neighbor_fn(index: PaperIndex):
    positions = {
        chunk.get("chunk_id"): i
        for i, chunk in enumerate(index.chunks)
        if chunk.get("chunk_id")
    }

    def neighbor_fn(chunk_id: str, window: int = 1) -> list[dict]:
        if chunk_id not in positions:
            return []
        pos = positions[chunk_id]
        width = max(1, window)
        start = max(0, pos - width)
        end = min(len(index.chunks), pos + width + 1)
        rows = []
        for i in range(start, end):
            chunk = index.chunks[i]
            if chunk.get("chunk_id") == chunk_id:
                continue
            rows.append({"chunk": chunk, "score": -abs(i - pos) / 100.0})
        return rows

    return neighbor_fn


def _fmt_agent_trace(rounds: list[dict]) -> list[dict]:
    trace = []
    for round_record in rounds:
        calls = []
        for result in round_record.get("tool_results", []):
            output = result.get("output", {}) or {}
            evidence_status = output.get("evidence_status", {}) or {}
            calls.append({
                "tool_name": result.get("tool_name", ""),
                "reason": result.get("args", {}).get("reason", ""),
                "observation": _compact_observation(result.get("tool_name", ""), output),
                "added_chunk_ids": output.get("added_chunk_ids", []),
                "evidence_count": evidence_status.get("evidence_count"),
                "suggested_next_actions": output.get("suggested_next_actions", [])[:3],
            })
        trace.append({
            "round": round_record.get("round"),
            "available_tools": round_record.get("available_tools", []),
            "calls": calls,
            "evidence_before": round_record.get("evidence_chunk_ids_before", []),
            "evidence_after": round_record.get("evidence_chunk_ids_after", []),
        })
    return trace


def _compact_observation(tool_name: str, output: dict) -> str:
    if not output.get("ok", False):
        return output.get("error", "Tool call failed.")
    if tool_name.startswith("retrieve"):
        added = len(output.get("added_chunk_ids", []))
        return f"Retrieved evidence and added {added} new chunk(s)."
    if tool_name == "rewrite_query":
        return f"Generated {len(output.get('rewrites', []))} rewrite(s)."
    if tool_name == "decompose_question":
        return f"Generated {len(output.get('subqueries', []))} subquery/subqueries."
    if tool_name == "list_neighbor_chunks":
        return f"Added {len(output.get('added_chunk_ids', []))} neighboring chunk(s)."
    if tool_name == "extract_table_data":
        return f"Extracted {output.get('tables_found', 0)} table(s)."
    if tool_name == "inspect_context":
        return "Evidence appears sufficient." if output.get("sufficient") else "Evidence gaps remain."
    if tool_name == "verify_answer_support":
        return (
            "Candidate answer verified against evidence."
            if output.get("verified")
            else f"{output.get('unsupported_claim_count', 0)} claim(s) still unsupported."
        )
    if tool_name == "finish":
        return output.get("reason", "Agent stopped.")
    return "Tool completed."


def _fmt_sources(retrieved: list[dict]) -> list[dict]:
    return [
        {
            "rank":       i + 1,
            "chunk_id":   r["chunk"]["chunk_id"],
            "section":    r["chunk"]["section"],
            "paper_title": r["chunk"].get("paper_title", ""),
            "paper_id":    r["chunk"].get("paper_id", ""),
            "text":       r["chunk"]["text"],
            "score":      round(r["score"], 4),
            "chunk_type": r["chunk"].get("chunk_type", "paragraph"),
            "numeric_facts": r.get("numeric_facts", []),
        }
        for i, r in enumerate(retrieved)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paperpilot.app:app", host="0.0.0.0", port=8000, reload=False)
