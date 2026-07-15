"""验证缺失元数据时软约束不会误删文档。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.retrieval.query.filter_parser import FilterParser
from coal_kb.retrieval.service import ExpertRetriever


class DummyRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def invoke(self, _query: str) -> list[Document]:
        return self._documents


def test_soft_scoring_missing_metadata_keeps_docs() -> None:
    documents = [
        Document(
            page_content="气化产生NH3",
            metadata={"chunk_id": "c1", "source_file": "a.pdf"},
        ),
        Document(
            page_content="热解产生酚类",
            metadata={"chunk_id": "c2", "source_file": "b.pdf"},
        ),
    ]

    def factory(k: int, where=None):
        return DummyRetriever(documents[:k])

    constraints = FilterParser(
        onto=Ontology.load("configs/schema.yaml")
    ).parse("1200K 气化 NH3")
    results = ExpertRetriever(
        vector_retriever_factory=factory,
        k=2,
    ).retrieve("1200K 气化 NH3", constraints)
    assert results
