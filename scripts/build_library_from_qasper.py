"""Build library index directly from QASPER JSON data.

Bypasses GROBID — uses the already-downloaded QASPER dev + train JSON
and pre-computed embedding caches for instant library setup.

Usage:
    python scripts/build_library_from_qasper.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

DEV_JSON   = Path("data/qasper_raw/qasper-dev-v0.3.json")
TRAIN_JSON = Path("data/qasper_raw/qasper-train-v0.3.json")
DEV_EMB    = Path("data/embeddings_mpnet.npy")
TRAIN_EMB  = Path("data/embeddings_mpnet_train.npy")
OUTPUT_DIR = Path("library_index")
MIN_CHARS  = 80


def load_papers(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_chunks(papers: dict, split_name: str) -> list[dict]:
    """Extract paragraph chunks from QASPER JSON, matching bench/chunking.py logic."""
    chunks = []
    for paper_id, paper in papers.items():
        title = paper.get("title", "")
        for sec_idx, section in enumerate(paper.get("full_text", [])):
            sec_name = section.get("section_name", "")
            for para_idx, para in enumerate(section.get("paragraphs", [])):
                if not para.strip() or len(para) < MIN_CHARS:
                    continue
                chunks.append({
                    "chunk_id":    f"{paper_id}__s{sec_idx}__p{para_idx}",
                    "paper_id":    paper_id,
                    "paper_title": title,
                    "section":     sec_name,
                    "chunk_type":  "paragraph",
                    "text":        para,
                    "index_text":  para,   # paragraphs: same as text
                })
    print(f"  {split_name}: {len({c['paper_id'] for c in chunks})} papers, "
          f"{len(chunks)} chunks")
    return chunks


def align_embeddings(chunks: list[dict], raw_emb: np.ndarray,
                     raw_papers: dict, split_name: str) -> np.ndarray:
    """Map cached embeddings to filtered chunks by position.

    The cached embeddings were built over ALL paragraphs (unfiltered).
    We rebuild the positional index to align them to our filtered chunks.
    """
    # Rebuild the same order as the cache (all paragraphs, no filter)
    pos = 0
    id_to_pos: dict[str, int] = {}
    for paper_id, paper in raw_papers.items():
        for sec_idx, section in enumerate(paper.get("full_text", [])):
            for para_idx, para in enumerate(section.get("paragraphs", [])):
                if para.strip():
                    cid = f"{paper_id}__s{sec_idx}__p{para_idx}"
                    id_to_pos[cid] = pos
                    pos += 1

    # Gather embeddings for the filtered chunks
    indices = []
    missing = 0
    for c in chunks:
        idx = id_to_pos.get(c["chunk_id"])
        if idx is not None and idx < len(raw_emb):
            indices.append(idx)
        else:
            indices.append(0)   # fallback: rare edge case
            missing += 1

    if missing:
        print(f"  Warning: {missing} chunks had no embedding match in {split_name}")

    return raw_emb[indices]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Load and chunk ────────────────────────────────────────────────────
    print("Loading QASPER papers…")
    dev_papers   = load_papers(DEV_JSON)
    train_papers = load_papers(TRAIN_JSON)

    print("Building chunks…")
    dev_chunks   = build_chunks(dev_papers,   "dev")
    train_chunks = build_chunks(train_papers, "train")
    all_chunks   = dev_chunks + train_chunks
    print(f"Total: {len(all_chunks)} chunks from "
          f"{len(dev_papers) + len(train_papers)} papers")

    # ── Align embeddings ──────────────────────────────────────────────────
    print("\nAligning pre-computed embeddings…")
    dev_emb_raw   = np.load(DEV_EMB)
    train_emb_raw = np.load(TRAIN_EMB)

    dev_embs   = align_embeddings(dev_chunks,   dev_emb_raw,   dev_papers,   "dev")
    train_embs = align_embeddings(train_chunks, train_emb_raw, train_papers, "train")
    embeddings = np.concatenate([dev_embs, train_embs], axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Save numpy embeddings (HNSW fallback) ─────────────────────────────
    emb_path = OUTPUT_DIR / "library_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ── Build HNSW (if hnswlib available) ─────────────────────────────────
    try:
        import hnswlib
        print("\nBuilding HNSW index…")
        dim   = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(all_chunks),
                         ef_construction=200, M=16)
        index.set_ef(50)
        index.add_items(embeddings, list(range(len(all_chunks))))
        hnsw_path = str(OUTPUT_DIR / "library_hnsw.bin")
        index.save_index(hnsw_path)
        print(f"Saved HNSW → {hnsw_path}")
    except ImportError:
        print("hnswlib not available — numpy fallback will be used")

    # ── Build papers metadata ──────────────────────────────────────────────
    papers_meta: dict = {}
    for paper_id, paper in {**dev_papers, **train_papers}.items():
        papers_meta[paper_id] = {
            "title":    paper.get("title", ""),
            "abstract": (paper.get("abstract") or "")[:500],
            "n_chunks": sum(1 for c in all_chunks if c["paper_id"] == paper_id),
        }

    with open(OUTPUT_DIR / "library_papers.json", "w", encoding="utf-8") as f:
        json.dump(papers_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved papers metadata → {OUTPUT_DIR / 'library_papers.json'}")

    # ── Save chunks (without embedding field) ─────────────────────────────
    with open(OUTPUT_DIR / "library_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"Saved chunks → {OUTPUT_DIR / 'library_chunks.json'}")

    # ── Build log ─────────────────────────────────────────────────────────
    build_log = {
        "n_papers":  len(papers_meta),
        "n_chunks":  len(all_chunks),
        "n_failed":  0,
        "source":    "QASPER dev + train JSON (no GROBID)",
        "build_time_s": round(time.time() - t0, 1),
        "hnsw_dim":  int(embeddings.shape[1]),
        "hnsw_M":    16,
        "hnsw_ef_construction": 200,
    }
    with open(OUTPUT_DIR / "build_log.json", "w", encoding="utf-8") as f:
        json.dump(build_log, f, indent=2)

    print(f"\nDone in {build_log['build_time_s']}s. "
          f"Library: {build_log['n_papers']} papers, {build_log['n_chunks']} chunks")
    print(f"Restart the server and open http://localhost:8000/library")


if __name__ == "__main__":
    main()
