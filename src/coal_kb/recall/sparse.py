"""提供中英文混合文本的候选集 BM25 召回。"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from langchain_core.documents import Document

from coal_kb.recall import config

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """按英文词、数字和单个中文字符切分，与旧实现保持一致。"""
    if not text:
        return []
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def bm25_rank(
    query: str,
    docs: Sequence[Document],
    *,
    k1: float = config.BM25_K1,
    b: float = config.BM25_B,
) -> list[tuple[Document, float]]:
    """仅在传入候选集上计算 BM25 排序。"""
    if not docs:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return [(doc, 0.0) for doc in docs]

    document_tokens = [tokenize(doc.page_content) for doc in docs]
    document_lengths = [len(tokens) for tokens in document_tokens]
    average_length = sum(document_lengths) / max(len(document_lengths), 1)

    document_frequency: dict[str, int] = {token: 0 for token in set(query_tokens)}
    for tokens in document_tokens:
        token_set = set(tokens)
        for token in document_frequency:
            if token in token_set:
                document_frequency[token] += 1

    document_count = len(docs)

    def _idf(term: str) -> float:
        matched_count = document_frequency.get(term, 0)
        return math.log((document_count - matched_count + 0.5) / (matched_count + 0.5) + 1.0)

    ranked: list[tuple[Document, float]] = []
    for doc, tokens, document_length in zip(docs, document_tokens, document_lengths):
        if document_length == 0:
            ranked.append((doc, 0.0))
            continue

        term_frequency: dict[str, int] = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + k1 * (1.0 - b + b * (document_length / average_length))
            score += _idf(term) * (frequency * (k1 + 1.0) / denominator)
        ranked.append((doc, float(score)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
