"""验证召回、检索、上下文和回答层的新旧入口保持兼容。"""

import pytest
from langchain_core.documents import Document

from coal_kb.answering import Answerer as CanonicalAnswerer
from coal_kb.answering.citations import build_rendered_citations, extract_referenced_labels
from coal_kb.answering.confidence import assess_evidence
from coal_kb.context import ContextBuilder as CanonicalContextBuilder
from coal_kb.context.builder import ContextBuilder as LegacyContextBuilder
from coal_kb.generation.answerer import Answerer as LegacyAnswerer
from coal_kb.recall import bm25_rank, rrf_fuse
from coal_kb.reranking import RerankingService
from coal_kb.retrieval.bm25 import bm25_rank as legacy_bm25_rank
from coal_kb.retrieval.bm25 import rrf_fuse as legacy_rrf_fuse
from coal_kb.retrieval.retriever import ExpertRetriever as LegacyExpertRetriever
from coal_kb.retrieval.service import ExpertRetriever as CanonicalExpertRetriever


def test_legacy_exports_point_to_canonical_layers() -> None:
    assert LegacyContextBuilder is CanonicalContextBuilder
    assert LegacyAnswerer is CanonicalAnswerer
    assert legacy_bm25_rank is bm25_rank
    assert legacy_rrf_fuse is rrf_fuse
    assert issubclass(LegacyExpertRetriever, CanonicalExpertRetriever)


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
