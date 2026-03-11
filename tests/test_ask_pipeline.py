from coal_kb.context.types import CitationItem
from coal_kb.generation.answerer import AnswerResult
from coal_kb.qa.ask_pipeline import AskExecution, build_response_payload, normalize_query, parse_command
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


def test_normalize_query_collapses_whitespace():
    assert normalize_query("  steam   gasification   NH3  ") == "steam gasification NH3"


def test_parse_command_supports_debug_and_exit():
    assert parse_command("debug") == "debug"
    assert parse_command("quit") == "exit"
    assert parse_command("steam gasification") is None


def test_build_response_payload_orders_referenced_labels_first():
    citations = {
        "E1": CitationItem(
            label="E1",
            source_file="paper-a.pdf",
            page=1,
            heading_path="Results",
            chunk_id="c1",
            snippet="A",
            source_display="paper-a.pdf | page 1 | Results",
        ).model_dump(),
        "E2": CitationItem(
            label="E2",
            source_file="paper-b.pdf",
            page=2,
            heading_path="Discussion",
            chunk_id="c2",
            snippet="B",
            source_display="paper-b.pdf | page 2 | Discussion",
        ).model_dump(),
    }
    execution = AskExecution(
        query="q",
        plan=_plan(),
        docs=[],
        trace={},
        context_debug={},
        result=AnswerResult(
            answer_text="Claim [E2]",
            citations=citations,
            used_chunks=["c1", "c2"],
            referenced_labels=["E2"],
            debug={},
        ),
        timings_ms={"total": 1.0},
    )
    payload = build_response_payload(execution)
    assert payload["citations"][0]["label"] == "E2"
    assert payload["citations"][0]["referenced_in_answer"] is True
