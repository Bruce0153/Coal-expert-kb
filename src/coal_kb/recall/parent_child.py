"""封装 Elasticsearch 父块到子块的两阶段召回。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from coal_kb.recall import config


@dataclass
class ParentChildRecallResult:
    """两阶段召回的原始候选与回退信息。"""

    parents: list[Document] = field(default_factory=list)
    children: list[Document] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    relax_steps: int = 0
    fallback_mode: str | None = None


@dataclass
class ParentChildRecall:
    """持有搜索存储和嵌入模型状态，执行原有父子块召回顺序。"""

    elastic_store: Any
    elastic_index: str
    embeddings: Any
    tenant_id: str | None = None
    use_icu: bool = False

    def process(
        self,
        query: str,
        *,
        where: dict[str, Any],
        parent_k_candidates: int,
        parent_k_final: int,
        max_parents: int,
        child_k_candidates: int,
        child_k_final: int,
        final_k: int,
        enable_relax: bool,
    ) -> ParentChildRecallResult:
        query_vector = self.embeddings.embed_query(query)

        parent_filters = dict(where)
        if self.tenant_id:
            parent_filters["tenant_id"] = self.tenant_id
        parents = self.elastic_store.search_parents(
            index=self.elastic_index,
            query_embedding=query_vector,
            query_text=query,
            filters=parent_filters,
            k_candidates=parent_k_candidates,
            k_final=parent_k_final,
            use_icu=self.use_icu,
        )
        parent_ids = [
            str((doc.metadata or {}).get("chunk_id"))
            for doc in parents
            if (doc.metadata or {}).get("chunk_id")
        ][:max_parents]
        parent_headings = {
            str((doc.metadata or {}).get("chunk_id")): str((doc.metadata or {}).get("heading_path") or "")
            for doc in parents
        }

        child_filters = dict(where)
        if self.tenant_id:
            child_filters["tenant_id"] = self.tenant_id
        if parent_ids:
            child_filters["parent_ids"] = parent_ids
        children = self.elastic_store.search_children(
            index=self.elastic_index,
            query_embedding=query_vector,
            query_text=query,
            filters=child_filters,
            k_candidates=child_k_candidates,
            k_final=child_k_final,
            use_icu=self.use_icu,
        )

        relax_steps = 0
        if not parent_ids or not children:
            relax_steps += 1
            fallback_filters: dict[str, Any] = {}
            if self.tenant_id:
                fallback_filters["tenant_id"] = self.tenant_id
            if enable_relax:
                fallback_filters = dict(where)
                fallback_filters.pop("T_range_K", None)
                fallback_filters.pop("P_range_MPa", None)
                if self.tenant_id:
                    fallback_filters["tenant_id"] = self.tenant_id
            children = self.elastic_store.search_children(
                index=self.elastic_index,
                query_embedding=query_vector,
                query_text=query,
                filters=fallback_filters,
                k_candidates=max(child_k_candidates, config.FALLBACK_CHILD_CANDIDATES),
                k_final=max(child_k_final, final_k, config.FALLBACK_CHILD_FINAL),
                use_icu=self.use_icu,
            )

        if not children:
            return ParentChildRecallResult(
                parents=list(parents),
                children=[],
                parent_ids=parent_ids,
                relax_steps=relax_steps + 1,
                fallback_mode="parent_as_evidence",
            )

        for doc in children:
            metadata = doc.metadata or {}
            parent_id = str(metadata.get("parent_id") or "")
            if parent_id in parent_headings:
                metadata["heading_path"] = parent_headings[parent_id]
            doc.metadata = metadata

        return ParentChildRecallResult(
            parents=list(parents),
            children=list(children),
            parent_ids=parent_ids,
            relax_steps=relax_steps,
        )
