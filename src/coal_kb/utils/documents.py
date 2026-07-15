"""提供跨检索、复杂问答和上下文层复用的文档工具。"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document


def document_key(document: Document) -> str:
    """返回优先使用 chunk_id 的稳定文档键。"""
    metadata = document.metadata or {}
    return str(
        metadata.get("chunk_id")
        or f"{metadata.get('source_file')}|{metadata.get('page')}|{document.page_content[:80]}"
    )


def copy_documents_with_metadata(
    documents: Iterable[Document],
    **metadata: object,
) -> list[Document]:
    """复制文档并追加元数据，避免修改召回缓存对象。"""
    output: list[Document] = []
    for document in documents:
        merged = dict(document.metadata or {})
        merged.update(metadata)
        output.append(Document(page_content=document.page_content, metadata=merged))
    return output


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
    """按稳定文档键保持原顺序去重。"""
    seen: set[str] = set()
    output: list[Document] = []
    for document in documents:
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output


def metadata_distribution(
    documents: Iterable[Document],
    key: str,
    *,
    default: str = "unknown",
) -> dict[str, int]:
    """统计文档元数据字段的值分布。"""
    distribution: dict[str, int] = {}
    for document in documents:
        value = str((document.metadata or {}).get(key) or default)
        distribution[value] = distribution.get(value, 0) + 1
    return distribution
