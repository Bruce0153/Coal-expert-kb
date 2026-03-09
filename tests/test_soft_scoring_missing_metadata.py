from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.metadata.normalize import Ontology
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever


class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _query: str):
        return self._docs


def test_soft_scoring_missing_metadata_keeps_docs() -> None:
    docs = [
        Document(page_content="气化产生NH3", metadata={"chunk_id": "c1", "source_file": "a.pdf"}),
        Document(page_content="热解产生酚类", metadata={"chunk_id": "c2", "source_file": "b.pdf"}),
    ]

    def factory(k: int, where=None):
        return DummyRetriever(docs[:k])

    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)
    constraints = parser.parse("1200K 气化 NH3")
    retriever = ExpertRetriever(vector_retriever_factory=factory, k=2, use_fuse=False)
    results = retriever.retrieve("1200K 气化 NH3", constraints)
    assert results
