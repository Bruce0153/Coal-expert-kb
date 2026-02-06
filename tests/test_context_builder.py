from langchain_core.documents import Document

from coal_kb.context.builder import ContextBuilder
from coal_kb.query.plan import (
    AnswerSpec,
    ContextSpec,
    DiversitySpec,
    NeighborSpec,
    ObservabilitySpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RerankSpec,
    RetrievalStep,
)


def _plan():
    return QueryPlan(
        query=QueryUnderstanding(raw="q", normalized="q"),
        retrieval_steps=[RetrievalStep(name="s", level="child", k_candidates=5, k_final=3)],
        relax_policy=RelaxPolicy(),
        rerank=RerankSpec(),
        neighbor=NeighborSpec(),
        diversity=DiversitySpec(max_per_source=2),
        context=ContextSpec(max_context_tokens=100, max_evidence_chunks=2, deduplicate=True),
        answer=AnswerSpec(),
        observability=ObservabilitySpec(trace_id="t"),
    )


def test_context_builder_budget_and_stable_citations():
    docs = [
        Document(page_content="A", metadata={"chunk_id": "c1", "source_file": "a.pdf", "heading_path": "H1", "page": 1}),
        Document(page_content="B", metadata={"chunk_id": "c2", "source_file": "a.pdf", "heading_path": "H1", "page": 2}),
        Document(page_content="C", metadata={"chunk_id": "c3", "source_file": "b.pdf", "heading_path": "H2", "page": 3}),
    ]
    pkg = ContextBuilder().build(_plan(), docs)
    assert len(pkg.used_chunks) == 2
    assert list(pkg.citations.keys()) == ["S1", "S2"]
    assert "## H1" in pkg.markdown
