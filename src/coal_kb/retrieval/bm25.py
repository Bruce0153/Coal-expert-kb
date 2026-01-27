from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document

# Tokenization for EN/CN mixed text:
# - english/number tokens
# - single CJK characters (simple but robust)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def bm25_rank(
    query: str,
    docs: Sequence[Document],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[Document, float]]:
    """
    BM25 ranking computed on the provided doc set (candidate set).
    This avoids building a global lexical index and keeps the pipeline simple & robust.
    """
    if not docs:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return [(d, 0.0) for d in docs]

    doc_tokens = [tokenize(d.page_content) for d in docs]
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)

    # document frequency for query terms
    df: Dict[str, int] = {t: 0 for t in set(q_tokens)}
    for toks in doc_tokens:
        s = set(toks)
        for t in df.keys():
            if t in s:
                df[t] += 1

    N = len(docs)

    def idf(term: str) -> float:
        # Robertson/Sparck Jones idf with + guaranteed positivity
        n_q = df.get(term, 0)
        return math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

    ranked: List[Tuple[Document, float]] = []
    for d, toks, dl in zip(docs, doc_tokens, doc_lens):
        if dl == 0:
            ranked.append((d, 0.0))
            continue

        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in q_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf(term) * (f * (k1 + 1.0) / denom)

        ranked.append((d, float(score)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def rrf_fuse(
    ranked_a: Sequence[Document],
    ranked_b: Sequence[Document],
    *,
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF).
    score(d) = sum( 1 / (k + rank(d)) )
    ranks are 1-based.
    """
    def key(d: Document) -> str:
        m = d.metadata or {}
        return str(m.get("chunk_id") or f'{m.get("source_file","")}|{m.get("page","")}|{d.page_content[:60]}')

    score: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, d in enumerate(ranked_a, start=1):
        kd = key(d)
        doc_map[kd] = d
        score[kd] = score.get(kd, 0.0) + 1.0 / (k + rank)

    for rank, d in enumerate(ranked_b, start=1):
        kd = key(d)
        doc_map[kd] = d
        score[kd] = score.get(kd, 0.0) + 1.0 / (k + rank)

    fused = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[kd] for kd, _ in fused]
