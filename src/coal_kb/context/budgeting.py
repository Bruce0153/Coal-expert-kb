"""负责上下文 token 估算和证据预算选择。"""

from __future__ import annotations

import re

from langchain_core.documents import Document


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", stripped, flags=re.UNICODE)))


def select_with_budget(
    docs: list[Document],
    *,
    max_chunks: int,
    max_tokens: int,
) -> tuple[list[Document], int, int]:
    selected: list[Document] = []
    dropped_budget = 0
    token_total = 0
    for doc in docs:
        if len(selected) >= max_chunks:
            dropped_budget += 1
            continue
        document_tokens = estimate_tokens(doc.page_content or "")
        if selected and max_tokens and token_total + document_tokens > max_tokens:
            dropped_budget += 1
            continue
        selected.append(doc)
        token_total += document_tokens
    return selected, dropped_budget, token_total
