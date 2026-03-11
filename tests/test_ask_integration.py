from langchain_core.documents import Document

from coal_kb.context.builder import ContextBuilder
from coal_kb.generation.answerer import Answerer
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
        diversity=DiversitySpec(),
        context=ContextSpec(),
        answer=AnswerSpec(),
        observability=ObservabilitySpec(trace_id="t"),
    )


def test_answerer_with_context_package_not_crash():
    pkg = ContextBuilder().build(_plan(), [])
    out = Answerer().answer(_plan(), pkg)
    assert "Insufficient evidence" in out.answer_text
    assert out.evidence_sufficiency == "insufficient"


def test_answerer_fallback_returns_claims_and_rendered_citations():
    docs = [
        Document(page_content="Steam gasification at 1200 K increases NH3.", metadata={"chunk_id": "c1", "source_file": "paper-a.pdf", "heading_path": "Results", "page": 4}),
        Document(page_content="HCN remains significant under the same conditions.", metadata={"chunk_id": "c2", "source_file": "paper-b.pdf", "heading_path": "Discussion", "page": 7}),
    ]
    pkg = ContextBuilder().build(_plan(), docs)
    out = Answerer().answer(_plan(), pkg, enable_llm=False)
    assert "[E1]" in out.answer_text
    assert out.referenced_labels == ["E1", "E2"]
    assert out.claim_items
    assert out.rendered_citations[0].startswith("[E1]")
    assert out.source_cards
