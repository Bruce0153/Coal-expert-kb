from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.context.builder import ContextBuilder
from coal_kb.query.plan import QueryPlan


def test_context_builder_budget_and_citations() -> None:
    builder = ContextBuilder()
    plan = QueryPlan(user_query="q", query_text="q", token_budget=50)
    docs = [
        Document(page_content="a" * 20, metadata={"heading_path": "H1", "source_file": "a.pdf", "chunk_id": "c1"}),
        Document(page_content="b" * 20, metadata={"heading_path": "H2", "source_file": "b.pdf", "chunk_id": "c2"}),
    ]
    pkg = builder.build(plan, docs)
    assert pkg.used_docs
    assert pkg.citations
    assert all(k.startswith("S") for k in pkg.citations)
