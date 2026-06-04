from difflib import SequenceMatcher
from bench.config import JACCARD_THRESHOLD


def filter_noise_chunks(chunks, qrels, min_chars=80):
    """Remove sub-min_chars chunks that are not referenced by any qrel.

    Must be called AFTER build_qrels so gold chunk ids are known.
    Gold chunks are preserved regardless of length to avoid
    silently dropping valid evidence from the evaluation set.
    """
    gold_ids = {cid for gold_set in qrels.values() for cid in gold_set}
    before = len(chunks)
    filtered = [
        c for c in chunks
        if len(c["text"]) >= min_chars or c["chunk_id"] in gold_ids
    ]
    short_gold = sum(
        1 for c in filtered
        if c["chunk_id"] in gold_ids and len(c["text"]) < min_chars
    )
    print(
        f"Noise filter: removed {before - len(filtered)} non-gold chunks "
        f"< {min_chars} chars ({short_gold} short gold chunks preserved)"
    )
    return filtered


def build_chunks(papers):
    chunks = []
    for paper in papers:
        paper_id = paper["id"]
        # full_text is a list of {section_name, paragraphs}
        for sec_idx, section_data in enumerate(paper["full_text"]):
            section = section_data["section_name"]
            for para_idx, para_text in enumerate(section_data["paragraphs"]):
                if not para_text.strip():
                    continue
                chunk_id = f"{paper_id}__s{sec_idx}__p{para_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "section": section,
                    "sec_idx": sec_idx,
                    "text": para_text,
                })
    return chunks


def build_qrels(queries, chunks, threshold=JACCARD_THRESHOLD):
    paper_chunks = {}
    for chunk in chunks:
        paper_chunks.setdefault(chunk["paper_id"], []).append(chunk)

    qrels = {}
    skipped = 0

    for query in queries:
        qid = query["qid"]
        paper_id = query["paper_id"]
        candidates = paper_chunks.get(paper_id, [])
        gold_chunks = set()

        for evidence_text in query["evidence"]:
            best_score = 0
            best_chunk_id = None
            for chunk in candidates:
                score = SequenceMatcher(
                    None, evidence_text[:500], chunk["text"][:500]
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk["chunk_id"]
            if best_score >= threshold:
                gold_chunks.add(best_chunk_id)

        if not gold_chunks:
            skipped += 1
            continue
        qrels[qid] = gold_chunks

    print(f"Skipped {skipped} queries (no chunk alignment above threshold={threshold})")
    print(f"Built qrels for {len(qrels)} queries")
    return qrels
