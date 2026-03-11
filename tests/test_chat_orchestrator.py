from coal_kb.chat.orchestrator import ChatOrchestrator
from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore
from coal_kb.generation.answerer import AnswerResult
from coal_kb.qa.ask_pipeline import AskExecution
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


class DummyRuntime:
    registry = None
    backend = "elastic"
    mode = "balanced"
    cfg = type("Cfg", (), {"model_versions": type("MV", (), {"embedding_version": "v1"})()})()
    retriever = type("Retriever", (), {"rerank_enabled": True})()


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


def test_chat_orchestrator_returns_conversation_capable_response(tmp_path, monkeypatch):
    service = ConversationService(ConversationStore(str(tmp_path / "chat.db")))

    def fake_execute_query(*args, **kwargs):
        return AskExecution(
            query="What about CO2 instead?",
            retrieval_query=args[1],
            plan=_plan(),
            docs=[],
            trace={},
            context_debug={},
            result=AnswerResult(
                answer_text="## Answer\nCO2 shifts the condition set [E1].",
                citations={
                    "E1": {
                        "label": "E1",
                        "source_file": "paper.pdf",
                        "title": "paper.pdf",
                        "page": 2,
                        "heading_path": "Results",
                        "chunk_id": "c1",
                        "snippet": "CO2 example",
                        "source_display": "paper.pdf | page 2 | Results",
                        "source_id": "paper|paper.pdf",
                    }
                },
                used_chunks=["c1"],
                evidence_items=[
                    {
                        "label": "E1",
                        "source_file": "paper.pdf",
                        "title": "paper.pdf",
                        "page": 2,
                        "heading_path": "Results",
                        "chunk_id": "c1",
                        "snippet": "CO2 example",
                        "source_display": "paper.pdf | page 2 | Results",
                        "source_id": "paper|paper.pdf",
                    }
                ],
                source_cards=[
                    {
                        "source_id": "paper|paper.pdf",
                        "source_file": "paper.pdf",
                        "title": "paper.pdf",
                        "pages": [2],
                        "headings": ["Results"],
                        "evidence_labels": ["E1"],
                        "evidence_count": 1,
                        "snippet_preview": "CO2 example",
                    }
                ],
                claim_items=[{"claim_id": "C1", "text": "CO2 shifts the condition set.", "citations": ["E1"], "support": "direct"}],
                rendered_citations=["[E1] paper.pdf | page 2 | Results"],
                referenced_labels=["E1"],
                evidence_sufficiency="limited",
                confidence_score=0.7,
                debug={},
            ),
            timings_ms={"total": 1.0},
            history_used=True,
            history_reason="follow_up_rewrite",
        )

    monkeypatch.setattr("coal_kb.chat.orchestrator.execute_query", fake_execute_query)
    monkeypatch.setattr("coal_kb.chat.orchestrator.log_query", lambda *args, **kwargs: None)

    orchestrator = ChatOrchestrator(conversations=service, runtime=DummyRuntime())
    first = orchestrator.chat(query="How does steam gasification affect NH3?", enable_llm=False)
    second = orchestrator.chat(
        query="What about CO2 instead?",
        conversation_id=first.conversation.conversation_id,
        enable_llm=False,
    )

    assert second.response["conversation_id"] == first.conversation.conversation_id
    assert second.response["message_id"]
    assert second.response["retrieval_trace_summary"]["history_used"] is True
    assert second.response["claim_items"][0]["citations"] == ["E1"]
