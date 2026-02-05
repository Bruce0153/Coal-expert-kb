from __future__ import annotations

from coal_kb.context.builder import ContextPackage
from coal_kb.generation.answerer import Answerer


def test_answerer_uncertain_on_empty_context() -> None:
    ans = Answerer(enable_llm=False, llm_provider="none")
    result = ans.answer("question", ContextPackage(prompt_context_text="", citations={}, used_docs=[], token_estimate=0))
    assert result.uncertain is True


def test_answerer_extractive_contains_citations() -> None:
    ans = Answerer(enable_llm=False, llm_provider="none")
    pkg = ContextPackage(
        prompt_context_text="",
        citations={"S1": {"source_file": "a.pdf", "heading_path": "H"}},
        used_docs=[type("D", (), {"page_content": "x", "metadata": {"citation_id": "S1"}})()],
        token_estimate=10,
    )
    result = ans.answer("question", pkg)
    assert "[S1]" in result.answer_text
