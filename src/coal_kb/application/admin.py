"""编排知识库文档上传、统计、删除和摄入操作。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text as sql_text

from coal_kb.infra.config import AppConfig
from coal_kb.infra.persistence.registry import RegistrySQLite
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.infra.security import build_upload_path


class AdminService:
    """持有配置与 Registry 状态，执行知识库管理用例。"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.registry = RegistrySQLite(cfg.registry.sqlite_path)

    def get_stats(self) -> dict[str, Any]:
        with self.registry._engine.connect() as connection:
            active_documents = connection.execute(
                sql_text("SELECT COUNT(*) FROM documents WHERE status = 'active'")
            ).scalar()
            total_documents = connection.execute(sql_text("SELECT COUNT(*) FROM documents")).scalar()
            total_chunks = connection.execute(sql_text("SELECT COUNT(*) FROM chunks")).scalar()
            last_ingestion = connection.execute(
                sql_text(
                    "SELECT finished_at FROM ingestion_runs WHERE status = 'completed' "
                    "ORDER BY finished_at DESC LIMIT 1"
                )
            ).scalar()
        return {
            "total_documents": total_documents or 0,
            "active_documents": active_documents or 0,
            "total_chunks": total_chunks or 0,
            "last_ingestion": str(last_ingestion) if last_ingestion else None,
            "backend": self.cfg.backend,
            "embedding_model": self.cfg.embeddings.model,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self.registry._engine.connect() as connection:
            rows = connection.execute(
                sql_text(
                    "SELECT document_id, source_file, title, doc_type, status, size, created_at "
                    "FROM documents ORDER BY created_at DESC LIMIT 200"
                )
            ).fetchall()
        return [
            {
                "document_id": row[0],
                "source_file": row[1],
                "title": row[2],
                "doc_type": row[3],
                "status": row[4],
                "size": row[5] or 0,
                "created_at": str(row[6]) if row[6] else "",
            }
            for row in rows
        ]

    def save_uploaded_document(self, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        extension = Path(safe_name).suffix.lower()
        directory = self._raw_pdf_dir() if extension == ".pdf" else self._raw_docs_dir()
        destination = build_upload_path(directory, safe_name)
        destination.write_bytes(content)
        return destination.name

    def delete_document(self, document_id: str) -> bool:
        if self.registry.get_document(document_id) is None:
            return False
        self.registry.delete_chunks_by_document_id(document_id)
        self.registry.delete_by_document_id(document_id)
        self._delete_from_chroma(document_id)
        self._delete_from_elasticsearch(document_id)
        return True

    def run_ingestion(self, *, rebuild: bool = False, force: bool = False) -> dict[str, Any]:
        try:
            from coal_kb.ingestion.pipeline import IngestPipeline

            result = IngestPipeline(cfg=self.cfg).run(rebuild=rebuild, force=force)
            return {
                "status": "completed",
                "message": f"摄入完成: {result.get('documents', 0)} 文档, {result.get('chunks', 0)} 分块",
                "stats": result,
            }
        except Exception as exc:
            return {"status": "failed", "message": f"摄入失败: {exc}", "stats": None}

    def _raw_pdf_dir(self) -> Path:
        path = Path(self.cfg.paths.raw_pdfs_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _raw_docs_dir(self) -> Path:
        path = Path(self.cfg.paths.raw_docs_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _delete_from_chroma(self, document_id: str) -> None:
        if self.cfg.backend not in {"chroma", "both"}:
            return
        try:
            store = ChromaStore(
                persist_dir=self.cfg.paths.chroma_dir,
                collection_name=self.cfg.chroma.collection_name,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                embedding_model=self.cfg.embedding.model_name,
            )
            store.delete_where({"document_id": document_id})
        except Exception:
            pass

    def _delete_from_elasticsearch(self, document_id: str) -> None:
        if self.cfg.backend not in {"elastic", "both"}:
            return
        try:
            store = ElasticStore(
                host=self.cfg.elastic.host,
                verify_certs=self.cfg.elastic.verify_certs,
                timeout_s=self.cfg.elastic.timeout_s,
            )
            store.delete_by_document_id(self.cfg.elastic.alias_current, document_id)
        except Exception:
            pass
