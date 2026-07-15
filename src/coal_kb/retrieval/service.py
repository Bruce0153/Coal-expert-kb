"""编排约束、召回、软排序、重排和多样性处理。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from coal_kb.core.models.query import Constraint as PlanConstraint
from coal_kb.core.models.query import QueryPlan
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig, make_embeddings
from coal_kb.ingestion.chunking.sectioner import is_reference_like
from coal_kb.recall import DenseRecall, ParentChildRecall
from coal_kb.reranking import RerankingService
from coal_kb.retrieval import config
from coal_kb.retrieval.constraints import Constraint, ConstraintSet, build_plan
from coal_kb.utils.documents import document_key, metadata_distribution

logger = logging.getLogger(__name__)


def _range_overlaps(
    metadata: dict[str, Any],
    query_range: list[float] | None,
    *,
    point_key: str,
    minimum_key: str,
    maximum_key: str,
) -> bool:
    if query_range is None or len(query_range) < 2:
        return True
    query_minimum, query_maximum = map(float, query_range[:2])
    document_minimum = metadata.get(minimum_key)
    document_maximum = metadata.get(maximum_key)
    if document_minimum is not None and document_maximum is not None:
        return max(float(document_minimum), query_minimum) <= min(
            float(document_maximum),
            query_maximum,
        )
    point = metadata.get(point_key)
    return point is not None and query_minimum <= float(point) <= query_maximum


@dataclass
class ExpertRetriever:
    """执行唯一正式约束模型驱动的检索流程。"""

    vector_retriever_factory: Any
    k: int = 6
    rerank_enabled: bool = False
    rerank_top_n: int = 10
    reranker: Any | None = None
    max_per_source: int = 2
    max_relax_steps: int = 2
    range_expand_schedule: list[float] | None = None
    mode: str = "balanced"
    drop_sections: list[str] | None = None
    drop_reference_like: bool = True
    two_stage_enabled: bool = True
    parent_k_candidates: int = 200
    parent_k_final: int = 60
    max_parents: int = 60
    child_k_candidates: int = 300
    child_k_final: int = 30
    allow_relax_in_stage2: bool = True
    elastic_store: ElasticStore | None = None
    elastic_index: str | None = None
    embeddings_cfg: EmbeddingsConfig | None = None
    elastic_use_icu: bool = False
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        self._dense_recall = DenseRecall(self.vector_retriever_factory)
        self._parent_child_recall: ParentChildRecall | None = None
        self._reranking_service = (
            RerankingService(self.reranker)
            if self.reranker is not None
            else None
        )
        if (
            self.two_stage_enabled
            and self.elastic_store is not None
            and self.embeddings_cfg is not None
            and self.elastic_index
        ):
            self._parent_child_recall = ParentChildRecall(
                elastic_store=self.elastic_store,
                elastic_index=self.elastic_index,
                embeddings=make_embeddings(self.embeddings_cfg),
                tenant_id=self.tenant_id,
                use_icu=self.elastic_use_icu,
            )

    def execute(
        self,
        plan: QueryPlan,
        trace: dict[str, Any] | None = None,
    ) -> list[Document]:
        """按照 QueryPlan 执行检索。"""
        query = plan.query.rewritten or plan.query.normalized
        constraints = self._constraints_from_plan(plan)
        if not self._two_stage_available():
            return self._retrieve_single_stage(query, constraints, trace)

        parent_step = next(
            (step for step in plan.retrieval_steps if step.level == "parent"),
            None,
        )
        child_step = next(
            (step for step in plan.retrieval_steps if step.level == "child"),
            None,
        )
        if parent_step is None or child_step is None:
            return self._retrieve_single_stage(query, constraints, trace)

        return self._retrieve_two_stage(
            query,
            constraints,
            trace,
            parent_k_candidates=parent_step.k_candidates,
            parent_k_final=parent_step.k_final,
            child_k_candidates=child_step.k_candidates,
            child_k_final=max(child_step.k_final, self.k),
            enable_relax=child_step.enable_relax,
            soft_constraints=[
                self._to_retrieval_constraint(constraint)
                for constraint in plan.query.soft_constraints
            ],
            plan=plan,
        )

    def retrieve(
        self,
        query: str,
        constraints: ConstraintSet,
        trace: dict[str, Any] | None = None,
    ) -> list[Document]:
        """使用正式 ConstraintSet 执行直接检索。"""
        if not self._two_stage_available():
            return self._retrieve_single_stage(query, constraints, trace)
        return self._retrieve_two_stage(
            query,
            constraints,
            trace,
            parent_k_candidates=self.parent_k_candidates,
            parent_k_final=self.parent_k_final,
            child_k_candidates=self.child_k_candidates,
            child_k_final=max(self.child_k_final, self.k),
            enable_relax=self.allow_relax_in_stage2,
        )

    def _two_stage_available(self) -> bool:
        return self.two_stage_enabled and self._parent_child_recall is not None

    def _retrieve_two_stage(
        self,
        query: str,
        constraints: ConstraintSet,
        trace: dict[str, Any] | None,
        *,
        parent_k_candidates: int,
        parent_k_final: int,
        child_k_candidates: int,
        child_k_final: int,
        enable_relax: bool,
        soft_constraints: list[Constraint] | None = None,
        plan: QueryPlan | None = None,
    ) -> list[Document]:
        recall = self._parent_child_recall
        if recall is None:
            raise RuntimeError("Two-stage retrieval is not configured")

        result = recall.process(
            query,
            where=constraints.to_where_hard(),
            parent_k_candidates=parent_k_candidates,
            parent_k_final=parent_k_final,
            max_parents=self.max_parents,
            child_k_candidates=child_k_candidates,
            child_k_final=child_k_final,
            final_k=self.k,
            enable_relax=enable_relax,
        )
        if result.fallback_mode == "parent_as_evidence":
            documents = result.parents[: self.k]
            if trace is not None:
                trace.update(
                    {
                        "stage1_parent_hits": len(result.parents),
                        "stage1_parent_ids": result.parent_ids,
                        "stage2_hits": 0,
                        "relax_steps": result.relax_steps,
                        "postfiltered_count": len(documents),
                        "fallback_mode": result.fallback_mode,
                    }
                )
            return documents

        ranking_constraints = soft_constraints
        if ranking_constraints is None:
            ranking_constraints = build_plan(
                constraints,
                max_relax_steps=self.max_relax_steps,
                range_expand_schedule=(
                    self.range_expand_schedule
                    or config.DEFAULT_RANGE_EXPAND_SCHEDULE
                ),
            ).soft_constraints

        filtered, scores = self._soft_rank(result.children, ranking_constraints)
        filtered = self._rerank(query, filtered)
        documents = self._apply_diversity(filtered)[: self.k]
        if trace is not None:
            trace.update(
                {
                    "stage1_parent_hits": len(result.parents),
                    "stage1_parent_ids": result.parent_ids,
                    "stage2_hits": len(result.children),
                    "relax_steps": result.relax_steps,
                    "postfiltered_count": len(filtered),
                    "condition_score_top3": [
                        {
                            "chunk_id": (document.metadata or {}).get("chunk_id"),
                            "score": scores.get(document_key(document), 0.0),
                        }
                        for document in filtered[:3]
                    ],
                    "final_top_citations": [
                        self._format_citation(document)
                        for document in documents[:3]
                    ],
                    "source_distribution": metadata_distribution(
                        documents,
                        "source_file",
                    ),
                    "heading_distribution": metadata_distribution(
                        documents,
                        "heading_path",
                    ),
                }
            )
            if plan is not None:
                trace["plan"] = plan.to_dict()
            if result.relax_steps:
                trace["two_stage_fallback"] = True
        return documents

    def _retrieve_single_stage(
        self,
        query: str,
        constraints: ConstraintSet,
        trace: dict[str, Any] | None,
    ) -> list[Document]:
        plan = build_plan(
            constraints,
            max_relax_steps=self.max_relax_steps,
            range_expand_schedule=(
                self.range_expand_schedule
                or config.DEFAULT_RANGE_EXPAND_SCHEDULE
            ),
        )
        documents = self._dense_recall.process(
            query,
            k=self.k,
            where=constraints.to_where_hard(),
        )
        if not documents:
            return []

        filtered, scores = self._soft_rank(documents, plan.soft_constraints)
        filtered = self._rerank(query, filtered)
        selected = self._apply_diversity(filtered)[: self.k]
        if trace is not None:
            trace.update(
                {
                    "where": constraints.to_where_hard(),
                    "vector_candidates": len(documents),
                    "postfiltered_count": len(filtered),
                    "condition_score_top3": [
                        {
                            "chunk_id": (document.metadata or {}).get("chunk_id"),
                            "score": scores.get(document_key(document), 0.0),
                        }
                        for document in filtered[:3]
                    ],
                    "final_top_citations": [
                        self._format_citation(document)
                        for document in selected[:3]
                    ],
                    "source_distribution": metadata_distribution(
                        selected,
                        "source_file",
                    ),
                    "heading_distribution": metadata_distribution(
                        selected,
                        "heading_path",
                    ),
                }
            )
        return selected

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if (
            not self.rerank_enabled
            or not documents
            or self._reranking_service is None
        ):
            return documents
        return self._reranking_service.process(
            query,
            documents,
            top_k=max(self.k, self.rerank_top_n),
        )

    @staticmethod
    def _to_retrieval_constraint(constraint: PlanConstraint) -> Constraint:
        constraint_type = (
            constraint.op
            if constraint.op in {"range", "enum", "set", "text"}
            else "enum"
        )
        return Constraint(
            name=constraint.field,
            ctype=constraint_type,
            value=constraint.value,
            confidence=constraint.confidence,
            source=constraint.source,
            priority=constraint.priority,
        )

    def _constraints_from_plan(self, plan: QueryPlan) -> ConstraintSet:
        return ConstraintSet(
            constraints=[
                self._to_retrieval_constraint(constraint)
                for constraint in (
                    plan.query.hard_constraints + plan.query.soft_constraints
                )
            ]
        )

    @staticmethod
    def _format_citation(document: Document) -> str:
        metadata = document.metadata or {}
        source = metadata.get("source_file", "unknown")
        heading = metadata.get("heading_path")
        chunk_id = metadata.get("chunk_id", "")
        if heading:
            return f"{source} [{heading}] #{chunk_id}"
        return f"{source} #{chunk_id}"

    def _soft_rank(
        self,
        documents: list[Document],
        constraints: list[Constraint],
    ) -> tuple[list[Document], dict[str, float]]:
        drop_sections = {
            section.lower().strip()
            for section in (self.drop_sections or [])
        }
        scores: dict[str, float] = {}
        kept: list[Document] = []
        for index, document in enumerate(documents):
            metadata = document.metadata or {}
            section = str(metadata.get("section", "unknown")).lower().strip()
            if section in drop_sections:
                continue
            if self.drop_reference_like and is_reference_like(document.page_content or ""):
                continue

            score = sum(
                value
                for constraint in constraints
                if (value := self._constraint_score(metadata, constraint)) > 0
            )
            score += 1.0 / (index + 1)
            scores[document_key(document)] = score
            kept.append(document)
        return (
            sorted(
                kept,
                key=lambda document: scores.get(document_key(document), 0.0),
                reverse=True,
            ),
            scores,
        )

    @staticmethod
    def _constraint_score(
        metadata: dict[str, Any],
        constraint: Constraint,
    ) -> float:
        weight = max(0.1, constraint.confidence)
        if constraint.ctype == "range":
            is_temperature = constraint.name == "T_range_K"
            return (
                weight
                if _range_overlaps(
                    metadata,
                    constraint.value,
                    point_key="T_K" if is_temperature else "P_MPa",
                    minimum_key="T_min_K" if is_temperature else "P_min_MPa",
                    maximum_key="T_max_K" if is_temperature else "P_max_MPa",
                )
                else 0.0
            )
        if constraint.ctype == "enum":
            return (
                weight
                if str(metadata.get(constraint.name, "")).lower()
                == str(constraint.value).lower()
                else 0.0
            )
        if constraint.ctype == "set":
            values = constraint.value or []
            hits = sum(
                1
                for value in values
                if metadata.get(
                    f"has_{value}"
                    if constraint.name == "targets"
                    else f"gas_{str(value).lower()}"
                )
            )
            return (hits / max(len(values), 1)) * weight if hits else 0.0
        if constraint.ctype == "text":
            return (
                0.5 * weight
                if str(constraint.value).lower()
                in str(metadata.get(constraint.name) or "").lower()
                else 0.0
            )
        return 0.0

    def _apply_diversity(
        self,
        documents: list[Document],
        max_per_source: int | None = None,
    ) -> list[Document]:
        limit = self.max_per_source if max_per_source is None else max_per_source
        if not documents or limit <= 0:
            return documents
        counts: dict[str, int] = {}
        output: list[Document] = []
        for document in documents:
            source = str((document.metadata or {}).get("source_file", "unknown"))
            if counts.get(source, 0) >= limit:
                continue
            counts[source] = counts.get(source, 0) + 1
            output.append(document)
        return output
