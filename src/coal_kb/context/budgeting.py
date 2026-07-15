"""负责上下文真实 Token 计数和证据预算选择。"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

TokenCounter = Callable[[str], int]


def select_with_budget(
    docs: list[Document],
    *,
    max_chunks: int,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> tuple[list[Document], int, int]:
    """使用当前模型 Tokenizer 选择预算内证据。"""
    selected: list[Document] = []
    dropped_budget = 0
    token_total = 0
    for doc in docs:
        if len(selected) >= max_chunks:
            dropped_budget += 1
            continue
        document_tokens = count_tokens(doc.page_content or "")
        if selected and max_tokens and token_total + document_tokens > max_tokens:
            dropped_budget += 1
            continue
        selected.append(doc)
        token_total += document_tokens
    return selected, dropped_budget, token_total
