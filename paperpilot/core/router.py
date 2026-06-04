"""Query intent router: determine if a query needs cross-table comparison.

Priority:
  1. Exact-match cache (0 ms) — avoids repeated LLM calls for identical queries
  2. LLM classifier (≈200 ms) — accurate intent detection via small model
  3. Keyword regex fallback — used when no LLM client is available

Cache is in-process (LRU, maxsize=256).  A production deployment would use
Redis with a short TTL; for the single-paper web app this is sufficient.
"""
import re
from functools import lru_cache

_CROSS_TABLE_PROMPT_SYSTEM = (
    "You are a query classifier for a research paper Q&A system.\n"
    "Determine whether the question requires jointly reading data from "
    "multiple distinct tables to answer (not just one table or a paragraph).\n"
    "Answer with exactly one word: yes or no."
)

_CROSS_TABLE_FALLBACK_RE = re.compile(
    r"(?i)\b(compared?\s+to|versus|vs\.?|higher\s+than|lower\s+than|"
    r"better\s+than|worse\s+than|outperform\w*|more\s+than|less\s+than|"
    r"difference\s+between|between\s+\S+\s+and)\b"
)


@lru_cache(maxsize=256)
def _cached_llm_route(question: str, model: str, base_url: str | None,
                      api_key_hint: str) -> bool:
    """Inner cached call — keyed on (question, model, base_url, api_key_hint).

    api_key_hint is the first 8 chars of the key so different keys produce
    different cache entries without storing the full secret.
    """
    import openai
    client = openai.OpenAI(api_key=_FULL_KEY, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CROSS_TABLE_PROMPT_SYSTEM},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_tokens=5,
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer.startswith("y")


# Module-level slot so the full key is available inside the cached function
_FULL_KEY: str = ""


def route_cross_table(question: str, llm_client=None) -> bool:
    """Return True if the query needs cross-table joint retrieval.

    Args:
        question:   User's question string.
        llm_client: An openai.OpenAI-compatible client.  When None, falls back
                    to keyword regex (instant but less accurate).
    """
    if llm_client is None:
        return bool(_CROSS_TABLE_FALLBACK_RE.search(question))

    import os
    global _FULL_KEY
    _FULL_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    model     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url  = os.environ.get("OPENAI_BASE_URL")
    key_hint  = _FULL_KEY[:8]

    try:
        return _cached_llm_route(question, model, base_url, key_hint)
    except Exception:
        # Network / quota failure → fall back to regex, never crash
        return bool(_CROSS_TABLE_FALLBACK_RE.search(question))
