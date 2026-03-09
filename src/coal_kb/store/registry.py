from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol


class Registry(Protocol):
    def upsert_document(
        self,
        *,
        document_id: str,
        source_file: str,
        sha256: str,
        mtime: int,
        size: int,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        parser: Optional[str] = None,
        title: Optional[str] = None,
        status: str = "active",
        tenant_id: Optional[str] = None,
    ) -> None: ...

    def upsert_chunks_bulk(self, chunks: Iterable[Dict[str, Any]]) -> None: ...

    def delete_by_document_id(self, document_id: str) -> None: ...

    def start_run(
        self,
        *,
        run_id: str,
        embedding_version: str,
        embedding_model: str,
        embedding_dim: int,
        schema_hash: str,
        chunking_signature: str,
    ) -> None: ...

    def finish_run(self, *, run_id: str, stats: Dict[str, Any], status: str) -> None: ...

    def log_query(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
        top_chunk_ids: Optional[list[str]],
        top_source_files: Optional[list[str]],
        latency_ms: Optional[float],
        backend: Optional[str],
        rerank_enabled: Optional[bool],
        tenant_id: Optional[str],
        embedding_version: Optional[str],
        mode: Optional[str] = None,
        relax_steps: Optional[list[str]] = None,
        diversity_k: Optional[int] = None,
    ) -> None: ...

    def log_run_metrics(
        self,
        *,
        run_id: str,
        index_name: str,
        embedding_version: str,
        schema_hash: str,
        doc_count: int,
        chunks: int,
        precision_at_k: Optional[float],
        recall_at_k: Optional[float],
        mrr: Optional[float],
    ) -> None: ...
