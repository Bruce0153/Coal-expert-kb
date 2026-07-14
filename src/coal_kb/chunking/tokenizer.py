"""兼容旧 token 计数导入路径，实际实现位于 tokenization.tokenizer。"""

from __future__ import annotations

from coal_kb.tokenization.tokenizer import (
    count_tokens,
)

__all__ = [
    "count_tokens",
]
