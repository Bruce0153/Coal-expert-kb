from __future__ import annotations

"""Token counting helpers for chunking.

This module prefers ``tiktoken`` for accurate token counting. When unavailable,
it falls back to a lightweight heuristic based on words/CJK characters/punctuation
so ingest does not fail in constrained environments.
"""

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return len(_TOKEN_RE.findall(text))
