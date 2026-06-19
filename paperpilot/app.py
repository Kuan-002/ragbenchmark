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
from paperpilot.core.parser        import parse_pdf, parse_arxiv
from paperpilot.core.classifier    import get_hint
from paperpilot.core.generator     import generate_answer
from paperpilot.core.router        import route_query

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
    return openai.OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
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
        return RedirectResponse("/?ui=en-only-2", status_code=307)
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


@app.post("/api/upload/arxiv")
async def upload_arxiv(arxiv_id: str = Form(...)):
    try:
        metadata, chunks = parse_arxiv(arxiv_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Download error: {e}")

    session_id = str(uuid.uuid4())
    log.info(f"Building index for arXiv {arxiv_id} ({len(chunks)} chunks)…")
    index = PaperIndex(chunks)
    sessions[session_id] = {"metadata": metadata, "index": index}

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
    mode:       str = "ce"
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

    if req.mode == "rrf":
        retrieved = index.retrieve_rrf(question, top_k=req.top_k)
    else:
        retrieved = index.retrieve_ce(question, top_k=req.top_k,
                                      cross_table=query_plan.cross_table,
                                      query_type=query_type)

    answer         = generate_answer(question, retrieved, query_plan.to_dict())
    low_confidence = bool(retrieved[0].get("low_confidence", False)) if retrieved else True

    return {
        "question":       question,
        "query_type":     query_type,
        "query_plan":     query_plan.to_dict(),
        "hint":           hint,
        "answer":         answer,
        "sources":        _fmt_sources(retrieved),
        "mode":           req.mode,
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

    metadata = data.get("metadata", {})
    chunks   = data.get("chunks", [])
    if not chunks:
        raise HTTPException(422, "Demo paper has no chunks")

    session_id = str(uuid.uuid4())
    log.info(f"Loading demo '{paper_key}' into session {session_id} ({len(chunks)} chunks)")
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


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    record = {
        "ts":           datetime.utcnow().isoformat(),
        "session_id":   req.session_id,
        "question":     req.question,
        "helpful":      req.helpful,
        "failure_type": req.failure_type,
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

    answer         = generate_answer(query, retrieved, query_plan.to_dict())
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

def _fmt_sources(retrieved: list[dict]) -> list[dict]:
    return [
        {
            "rank":       i + 1,
            "chunk_id":   r["chunk"]["chunk_id"],
            "section":    r["chunk"]["section"],
            "text":       r["chunk"]["text"],
            "score":      round(r["score"], 4),
            "chunk_type": r["chunk"].get("chunk_type", "paragraph"),
        }
        for i, r in enumerate(retrieved)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paperpilot.app:app", host="0.0.0.0", port=8000, reload=False)
