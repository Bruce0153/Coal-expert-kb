from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.retrieval.retriever import ExpertRetriever


class FakeEmb:
    def embed_query(self, q):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self):
        self.parent_filters = None
        self.child_filters = None

    def search_parents(self, **kwargs):
        self.parent_filters = kwargs["filters"]
        return [Document(page_content="p", metadata={"chunk_id": "p1", "heading_path": "M > E"})]

    def search_children(self, **kwargs):
        self.child_filters = kwargs["filters"]
        return [Document(page_content="c", metadata={"chunk_id": "c1", "parent_id": "p1", "source_file": "a.pdf"})]


def test_two_stage_filters_applied(monkeypatch):
    monkeypatch.setattr("coal_kb.retrieval.retriever.make_embeddings", lambda cfg: FakeEmb())
    store = FakeStore()
    r = ExpertRetriever(
        vector_retriever_factory=lambda k, where=None: None,
        k=1,
        two_stage_enabled=True,
        elastic_store=store,
        elastic_index="idx",
        embeddings_cfg=EmbeddingsConfig(provider="dashscope", base_url="http://localhost", api_key_env="DUMMY", model="dummy", dimensions=2),
        where_full=True,
    )
    docs = r.retrieve("q", {"stage": "gasification"})
    assert docs
    assert store.parent_filters["stage"] == "gasification"
    assert store.child_filters.get("parent_ids") == ["p1"]


def test_two_stage_fallback_when_no_parents(monkeypatch):
    monkeypatch.setattr("coal_kb.retrieval.retriever.make_embeddings", lambda cfg: FakeEmb())

    class EmptyParentStore(FakeStore):
        def search_parents(self, **kwargs):
            self.parent_filters = kwargs["filters"]
            return []

        def search_children(self, **kwargs):
            self.child_filters = kwargs["filters"]
            return [Document(page_content="c", metadata={"chunk_id": "c1", "parent_id": "pX", "source_file": "a.pdf"})]

    store = EmptyParentStore()
    r = ExpertRetriever(
        vector_retriever_factory=lambda k, where=None: None,
        k=1,
        two_stage_enabled=True,
        elastic_store=store,
        elastic_index="idx",
        embeddings_cfg=EmbeddingsConfig(provider="dashscope", base_url="http://localhost", api_key_env="DUMMY", model="dummy", dimensions=2),
        where_full=True,
    )
    trace = {}
    docs = r.retrieve("q", {}, trace=trace)
    assert docs
    assert trace.get("two_stage_fallback") is True
