"""Run the QASPER RAG benchmark.

Examples:
  python run_bench.py --max-papers 228 --llm-per-type 10
  python run_bench.py --max-papers 50 --no-llm
  python run_bench.py --max-papers 50 --no-llm --strategies RRF+CE Fusion+CE
  python run_bench.py --from-cache data/results_latest.json
"""
import argparse
import os

# Load .env if present.
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from bench.runner import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Limit number of papers (default: all 228)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM judge (Tier 2 & 3)")
    parser.add_argument("--llm-per-type", type=int, default=10,
                        help="LLM judge samples per query type")
    parser.add_argument("--from-cache", type=str, default=None,
                        help="Load cached results JSON and regenerate report only")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional stable run id under data/runs/")
    parser.add_argument("--output-root", type=str, default="data/runs",
                        help="Directory where immutable run artifacts are stored")
    parser.add_argument("--no-latest", action="store_true",
                        help="Do not update data/report.md and data/results_latest.json")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Optional subset, e.g. RRF+CE Fusion+CE")
    args = parser.parse_args()

    llm_client = None
    if not args.no_llm and args.from_cache is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            import openai
            base_url = os.environ.get("OPENAI_BASE_URL") or (
                "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
            )
            llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            print(f"LLM client initialised: model={model}, base_url={base_url}")
        else:
            print("No OPENAI_API_KEY / DEEPSEEK_API_KEY found; running without LLM judge.")
            args.no_llm = True

    run(
        max_papers=args.max_papers,
        no_llm=args.no_llm,
        llm_client=llm_client,
        from_cache=args.from_cache,
        run_id=args.run_id,
        output_root=args.output_root,
        preserve_latest=not args.no_latest,
        strategies_filter=args.strategies,
    )


if __name__ == "__main__":
    main()
