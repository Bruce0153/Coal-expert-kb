from __future__ import annotations

import os
import time

import pytest

from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.retrieval.elastic_retriever import ElasticRetriever
from coal_kb.store.elastic_store import ElasticStore


@pytest.mark.skipif(os.environ.get("RUN_ES_TESTS") != "1", reason="RUN_ES_TESTS not set")
def test_elastic_retriever_roundtrip(monkeypatch):
    elasticsearch = pytest.importorskip("elasticsearch")
    client = elasticsearch.Elasticsearch("http://localhost:9200", verify_certs=False)
    if not client.ping():
        pytest.skip("Elasticsearch not available at localhost:9200")

    store = ElasticStore(host="http://localhost:9200", verify_certs=False)
    index_name = f"coal_kb_test_{int(time.time())}"
    store.create_index(index_name, dims=4)

    docs = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "source_file": "a.pdf",
            "page": 1,
            "section": "results",
            "chunk_index": 0,
            "text": "Steam gasification produces NH3.",
            "stage": "gasification",
            "embedding": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "source_file": "b.pdf",
            "page": 2,
            "section": "results",
            "chunk_index": 1,
            "text": "Pyrolysis produces tar.",
            "stage": "pyrolysis",
            "embedding": [0.2, 0.1, 0.4, 0.3],
        },
    ]
    store.bulk_upsert_chunks(index_name, docs)
    client.indices.refresh(index=index_name)

    class DummyEmbeddings:
        def embed_query(self, _query: str):
            return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr(
        "coal_kb.retrieval.elastic_retriever.make_embeddings",
        lambda _cfg: DummyEmbeddings(),
    )

    retriever = ElasticRetriever(
        client=client,
        index=index_name,
        embeddings_cfg=EmbeddingsConfig(
            provider="openai",
            base_url="http://localhost",
            api_key_env="DUMMY",
            model="dummy",
            dimensions=4,
        ),
        k=2,
    )
    docs_out = retriever.invoke("steam gasification")
    assert docs_out
    assert any("gasification" in (d.page_content or "").lower() for d in docs_out)

    client.indices.delete(index=index_name)
