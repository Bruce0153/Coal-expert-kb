"""执行证据去重与来源轮询多样化。"""

from __future__ import annotations

from collections import OrderedDict, deque

from langchain_core.documents import Document

from coal_kb.context import config
from coal_kb.context.citations import snippet, source_id


def deduplicate_docs(docs: list[Document], *, enabled: bool) -> tuple[list[Document], int]:
    selected: list[Document] = []
    used_chunk_ids: set[str] = set()
    used_texts: set[str] = set()
    dropped_duplicates = 0
    for doc in docs:
        metadata = doc.metadata or {}
        chunk_id = str(metadata.get("chunk_id") or "")
        text_key = snippet(doc.page_content or "", max_chars=config.DEDUP_SNIPPET_CHARS).lower()
        if enabled:
            duplicate = bool(chunk_id and chunk_id in used_chunk_ids) or text_key in used_texts
            if duplicate:
                dropped_duplicates += 1
                continue
        selected.append(doc)
        if chunk_id:
            used_chunk_ids.add(chunk_id)
        used_texts.add(text_key)
    return selected, dropped_duplicates


def diversify_docs(docs: list[Document]) -> list[Document]:
    grouped: OrderedDict[str, deque[Document]] = OrderedDict()
    for doc in docs:
        grouped.setdefault(source_id(doc.metadata or {}), deque()).append(doc)

    diversified: list[Document] = []
    while grouped:
        empty_keys: list[str] = []
        for key, bucket in grouped.items():
            if bucket:
                diversified.append(bucket.popleft())
            if not bucket:
                empty_keys.append(key)
        for key in empty_keys:
            grouped.pop(key, None)
    return diversified
