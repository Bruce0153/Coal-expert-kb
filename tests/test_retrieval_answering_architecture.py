"""验证召回、检索、上下文和回答层的 canonical 边界。"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from coal_kb.answering import Answerer
from coal_kb.answering.citations import build_rendered_citations, extract_referenced_labels
from coal_kb.answering.confidence import assess_evidence
from coal_kb.context import ContextBuilder
from coal_kb.recall import bm25_rank, rrf_fuse
from coal_kb.reranking import RerankingService
from coal_kb.retrieval.service import ExpertRetriever

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def test_canonical_packages_export_primary_services() -> None:
    assert Answerer.__module__.startswith("coal_kb.answering")
    assert ContextBuilder.__module__ == "coal_kb.context.service"
    assert ExpertRetriever.__module__ == "coal_kb.retrieval.service"
    assert callable(bm25_rank)
    assert callable(rrf_fuse)


def test_reranking_service_preserves_remaining_order() -> None:
    docs = [Document(page_content=name, metadata={"chunk_id": name}) for name in ["A", "B", "C"]]

    class FakeReranker:
        def rerank(self, query, candidates, top_k):  # noqa: ANN001, ANN201
            assert query == "query"
            assert top_k == 2
            return list(reversed(candidates))

    result = RerankingService(FakeReranker()).process("query", docs, top_k=2)
    assert [doc.metadata["chunk_id"] for doc in result] == ["B", "A", "C"]


def test_answering_helpers_keep_existing_rules() -> None:
    assert assess_evidence(0, 2) == ("insufficient", 0.0)
    assert assess_evidence(1, 2) == ("insufficient", 0.25)
    assert assess_evidence(2, 2) == ("partial", 0.65)
    status, score = assess_evidence(4, 2)
    assert status == "sufficient"
    assert score == pytest.approx(0.78)

    labels = extract_referenced_labels("结论 [E2]，补充 [E1]，重复 [E2]。", ["E1", "E2"])
    assert labels == ["E2", "E1"]
    rendered = build_rendered_citations(
        {"E1": {"source_file": "a.pdf", "page": 2}, "E2": {"source_file": "b.pdf", "page": None}},
        labels,
    )
    assert rendered == ["[E2] b.pdf", "[E1] a.pdf (page 2)"]


def test_layer_does_not_recreate_removed_modules() -> None:
    removed = ["generation", "query", "qa", "eval"]
    assert all(not (ROOT / name).exists() for name in removed)
