"""验证两阶段检索使用统一约束模型。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.core.models.query import (
    AnswerSpec,
    Constraint,
    ContextSpec,
    DiversitySpec,
    NeighborSpec,
    ObservabilitySpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RelaxRule,
    RerankSpec,
    RetrievalStep,
)
from coal_kb.infra.providers.config import default_embeddings_config
from coal_kb.retrieval.constraints import Constraint as RetrievalConstraint
from coal_kb.retrieval.constraints import ConstraintSet
from coal_kb.retrieval.service import ExpertRetriever


def _plan() -> QueryPlan:
    return QueryPlan(
        query=QueryUnderstanding(
            raw="q",
            normalized="q",
            soft_constraints=[
                Constraint(
                    field="stage",
                    op="enum",
                    value="gasification",
                )
            ],
        ),
        retrieval_steps=[
            RetrievalStep(
                name="s1",
                level="parent",
                k_candidates=5,
                k_final=3,
            ),
            RetrievalStep(
                name="s2",
                level="child",
                k_candidates=5,
                k_final=3,
                enable_relax=True,
            ),
        ],
        relax_policy=RelaxPolicy(
            max_steps=1,
            rules=[RelaxRule(drop_fields=["T_range_K"])],
        ),
        rerank=RerankSpec(enabled=False),
        neighbor=NeighborSpec(enabled=False),
        diversity=DiversitySpec(max_per_source=2),
        context=ContextSpec(),
        answer=AnswerSpec(),
        observability=ObservabilitySpec(trace_id="t"),
    )


def _hard_stage_constraints() -> ConstraintSet:
    return ConstraintSet(
        constraints=[
            RetrievalConstraint(
                name="stage",
                ctype="enum",
                value="gasification",
                confidence=1.0,
                source="test",
                priority="hard",
            )
        ]
    )


class FakeEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2]


class FakeStore:
    def __init__(self) -> None:
        self.parent_filters = None
        self.child_filters = None

    def search_parents(self, **kwargs):
        self.parent_filters = kwargs["filters"]
        return [
            Document(
                page_content="p",
                metadata={"chunk_id": "p1", "heading_path": "M > E"},
            )
        ]

    def search_children(self, **kwargs):
        self.child_filters = kwargs["filters"]
        return [
            Document(
                page_content="c",
                metadata={
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "source_file": "a.pdf",
                },
            )
        ]


def _retriever(monkeypatch, store: FakeStore) -> ExpertRetriever:
    monkeypatch.setattr(
        "coal_kb.retrieval.service.make_embeddings",
        lambda _config: FakeEmbeddings(),
    )
    return ExpertRetriever(
        vector_retriever_factory=lambda k, where=None: None,
        k=1,
        two_stage_enabled=True,
        elastic_store=store,
        elastic_index="idx",
        embeddings_cfg=default_embeddings_config(),
    )


def test_two_stage_filters_applied(monkeypatch) -> None:
    store = FakeStore()
    documents = _retriever(monkeypatch, store).retrieve(
        "q",
        _hard_stage_constraints(),
    )
    assert documents
    assert store.parent_filters["stage"] == "gasification"
    assert store.child_filters.get("parent_ids") == ["p1"]


def test_two_stage_fallback_when_no_parents(monkeypatch) -> None:
    class EmptyParentStore(FakeStore):
        def search_parents(self, **kwargs):
            self.parent_filters = kwargs["filters"]
            return []

        def search_children(self, **kwargs):
            self.child_filters = kwargs["filters"]
            return [
                Document(
                    page_content="c",
                    metadata={
                        "chunk_id": "c1",
                        "parent_id": "pX",
                        "source_file": "a.pdf",
                    },
                )
            ]

    trace = {}
    documents = _retriever(monkeypatch, EmptyParentStore()).retrieve(
        "q",
        ConstraintSet(),
        trace=trace,
    )
    assert documents
    assert trace.get("two_stage_fallback") is True


def test_execute_uses_plan_parent_ids(monkeypatch) -> None:
    store = FakeStore()
    documents = _retriever(monkeypatch, store).execute(_plan(), trace={})
    assert documents
    assert store.child_filters.get("parent_ids") == ["p1"]


def test_execute_fallback_when_stage1_empty(monkeypatch) -> None:
    class EmptyParentStore(FakeStore):
        def search_parents(self, **kwargs):
            return []

    trace = {}
    documents = _retriever(monkeypatch, EmptyParentStore()).execute(
        _plan(),
        trace=trace,
    )
    assert documents
    assert trace.get("two_stage_fallback") is True
