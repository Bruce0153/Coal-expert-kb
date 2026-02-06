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


def test_answerer_with_context_package_not_crash():
    plan = QueryPlan(
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
    pkg = ContextBuilder().build(plan, [])
    out = Answerer().answer(plan, pkg)
    assert "无法可靠回答" in out.answer_text
