"""Pre-parse a paper and save it as a demo JSON file.

The output JSON is loaded at runtime without GROBID, so demo users
don't need Docker.

Usage:
    # From a local PDF
    python scripts/save_demo_paper.py --pdf papers/attention.pdf --key attention --name "Attention Is All You Need"

    # From arXiv ID
    python scripts/save_demo_paper.py --arxiv 1706.03762 --key attention --name "Attention Is All You Need"

Output: demo_papers/<key>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",   help="Local PDF path")
    src.add_argument("--arxiv", help="arXiv ID (e.g. 1706.03762)")
    parser.add_argument("--key",  required=True, help="Short key used as filename, e.g. attention")
    parser.add_argument("--name", required=True, help="Display name shown in the UI")
    parser.add_argument("--out",  default="demo_papers", help="Output directory")
    args = parser.parse_args()

    from paperpilot.core.parser import parse_pdf, parse_arxiv

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: file not found: {pdf_path.resolve()}")
            print("Tip:  put PDFs in the papers/ directory, then run:")
            print(f"      python scripts/save_demo_paper.py --pdf papers/<filename>.pdf ...")
            print("      Or use --arxiv <id> to download directly from arXiv.")
            sys.exit(1)

    print(f"Parsing {'arXiv:' + args.arxiv if args.arxiv else args.pdf} …")
    if args.arxiv:
        metadata, chunks = parse_arxiv(args.arxiv)
    else:
        metadata, chunks = parse_pdf(pdf_path.read_bytes())

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.key}.json"

    payload = {
        "key":          args.key,
        "display_name": args.name,
        "metadata":     metadata,
        "chunks":       chunks,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
