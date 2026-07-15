"""提供复杂路线复用的 QueryPlan 克隆、文档标记和去重函数。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.core.models.query import ComplexQuestionSpec, QueryPlan, SubQuerySpec


def clone_plan_for_subquery(plan: QueryPlan, subquery: SubQuerySpec) -> QueryPlan:
    """克隆计划并把复杂子问题降级为一次普通事实检索。"""
    cloned = plan.model_copy(deep=True)
    cloned.query.raw = subquery.query
    cloned.query.normalized = subquery.query
    cloned.query.rewritten = None
    cloned.complex = ComplexQuestionSpec(query_type="fact", confidence=1.0, reason="复杂路线内部子检索")
    return cloned


def tag_documents(documents: list[Document], **metadata: object) -> list[Document]:
    """复制文档并附加复杂路线元数据，避免修改召回缓存对象。"""
    tagged: list[Document] = []
    for document in documents:
        merged = dict(document.metadata or {})
        merged.update(metadata)
        tagged.append(Document(page_content=document.page_content, metadata=merged))
    return tagged


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """按 chunk_id 或来源页码稳定去重。"""
    seen: set[str] = set()
    output: list[Document] = []
    for document in documents:
        metadata = document.metadata or {}
        key = str(metadata.get("chunk_id") or f"{metadata.get('source_file')}|{metadata.get('page')}|{document.page_content[:80]}")
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output
