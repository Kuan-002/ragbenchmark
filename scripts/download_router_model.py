"""Download the local router embedding model into the project.

Default model:
  sentence-transformers/all-MiniLM-L6-v2

Output:
  models/router/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/router/all-MiniLM-L6-v2",
        help="Local directory where the model should be saved.",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    model.save(str(output_dir))
    print(f"Saved router model to {output_dir}")


if __name__ == "__main__":
    main()
