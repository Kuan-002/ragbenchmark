"""Offline script: parse PDFs → GROBID → embed → build HNSW + BM25 index.

Usage:
    python scripts/build_library_index.py \\
        --papers_dir ./papers/ \\
        --output_dir ./library_index/ \\
        --batch_size 32

Requires GROBID running at $GROBID_URL (default http://localhost:8070).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_dir",  default="./papers/")
    parser.add_argument("--output_dir",  default="./library_index/")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--min_chars",   type=int, default=80)
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs in {papers_dir}")

    # ── Parse with GROBID ──────────────────────────────────────────────────
    from paperpilot.core.parser import parse_pdf

    all_chunks:  list[dict] = []
    papers_meta: dict       = {}
    failed                  = []
    t_parse = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        paper_id = pdf_path.stem   # filename without .pdf
        print(f"  [{i:3d}/{len(pdf_files)}] {pdf_path.name} … ", end="", flush=True)
        try:
            meta, chunks = parse_pdf(pdf_path.read_bytes())
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(pdf_path.name)
            continue

        # Filter short chunks (already done in parser, but double-check)
        chunks = [c for c in chunks if len(c.get("text", "")) >= args.min_chars]

        for c in chunks:
            c["paper_id"]    = paper_id
            c["paper_title"] = meta.get("title", "")
            c["chunk_id"]    = f"{paper_id}_{c['chunk_id']}"

        all_chunks.extend(chunks)
        papers_meta[paper_id] = {
            "title":    meta.get("title", ""),
            "abstract": meta.get("abstract", ""),
            "n_chunks": len(chunks),
        }
        print(f"{len(chunks)} chunks")

    parse_time = time.time() - t_parse
    print(f"\nParsed {len(papers_meta)} papers, {len(all_chunks)} chunks "
          f"({len(failed)} failed) in {parse_time:.0f}s")

    if not all_chunks:
        print("No chunks to index. Exiting.")
        return

    # ── Embed with Dense model ─────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("\nEncoding embeddings …")
    model    = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    texts    = [c["text"] for c in all_chunks]
    t_embed  = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embed_time = time.time() - t_embed
    print(f"Encoded {len(embeddings)} chunks in {embed_time:.0f}s")

    # ── Save embeddings (numpy fallback, always) ───────────────────────────
    emb_path = output_dir / "library_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ── Build HNSW (optional, requires hnswlib) ────────────────────────────
    try:
        import hnswlib
        print("\nBuilding HNSW index …")
        dim   = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(all_chunks), ef_construction=200, M=16)
        index.set_ef(50)
        index.add_items(embeddings, list(range(len(all_chunks))))
        hnsw_path = str(output_dir / "library_hnsw.bin")
        index.save_index(hnsw_path)
        print(f"Saved HNSW → {hnsw_path}")
    except ImportError:
        print("hnswlib not installed — skipping HNSW. Numpy fallback will be used.")

    # ── Build BM25 (in-memory, rebuilt at server startup) ──────────────────
    print("BM25 will be built from chunks.json at server startup (< 5s).")

    # ── Save metadata ──────────────────────────────────────────────────────
    # Strip embedding field (not stored per chunk)
    chunks_out = []
    for c in all_chunks:
        entry = {k: v for k, v in c.items() if k != "embedding"}
        chunks_out.append(entry)

    chunks_path = output_dir / "library_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_out, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks → {chunks_path}")

    papers_path = output_dir / "library_papers.json"
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved papers → {papers_path}")

    build_log = {
        "n_papers":      len(papers_meta),
        "n_chunks":      len(all_chunks),
        "n_failed":      len(failed),
        "failed":        failed,
        "parse_time_s":  round(parse_time, 1),
        "embed_time_s":  round(embed_time, 1),
        "hnsw_dim":      dim,
        "hnsw_M":        16,
        "hnsw_ef_construction": 200,
    }
    log_path = output_dir / "build_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(build_log, f, indent=2)
    print(f"Saved build log → {log_path}")
    print(f"\nDone. Library index ready at {output_dir}")


if __name__ == "__main__":
    main()
