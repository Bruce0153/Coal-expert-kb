from __future__ import annotations

from pathlib import Path

from coal_kb.store.registry_sqlite import RegistrySQLite


def test_registry_upserts_and_deletes(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.db"
    registry = RegistrySQLite(str(db_path))

    registry.upsert_document(
        document_id="doc1",
        source_file="a.pdf",
        sha256="sha1",
        mtime=123,
        size=456,
        status="active",
    )

    registry.upsert_chunks_bulk(
        [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "page": 1,
                "section": "results",
                "chunk_index": 0,
                "text": "hello",
                "metadata_json": "{}",
                "embedding_model": "test",
                "embedding_dim": 4,
                "embedding_version": "v1",
            }
        ]
    )

    registry.log_query(
        query="steam gasification",
        filters={"stage": "gasification"},
        top_chunk_ids=["c1"],
        top_source_files=["a.pdf"],
        latency_ms=12.3,
        backend="chroma",
        tenant_id=None,
        embedding_version="v1",
        rerank_enabled=True,
    )

    registry.delete_chunks_by_document_id("doc1")
