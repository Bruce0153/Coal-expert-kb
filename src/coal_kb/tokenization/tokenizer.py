"""提供稳定的 token 计数并在依赖缺失时回退。"""

from __future__ import annotations

import re
from functools import lru_cache

from coal_kb.tokenization import config

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding(config.TIKTOKEN_ENCODING)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return len(_TOKEN_RE.findall(text))
