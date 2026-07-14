from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from coal_kb.ingestion.pipeline import IngestPipeline
from coal_kb.infra.config import AppConfig
from coal_kb.infra.persistence.registry import RegistrySQLite


class DocumentInfo(BaseModel):
    document_id: str
    source_file: str
    title: str | None = None
    doc_type: str | None = None
    status: str = "active"
    size: int = 0
    created_at: str = ""


class KBStats(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    active_documents: int = 0
    last_ingestion: str | None = None
    backend: str = ""
    embedding_model: str = ""


class IngestResult(BaseModel):
    status: str
    message: str
    stats: dict | None = None


def build_admin_router(cfg: AppConfig) -> APIRouter:
    router = APIRouter(prefix="/api/admin", tags=["admin"])
    registry = RegistrySQLite(cfg.registry.sqlite_path)

    def _raw_pdf_dir() -> Path:
        p = Path(cfg.paths.raw_pdfs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _raw_docs_dir() -> Path:
        p = Path(cfg.paths.raw_docs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @router.get("/stats", response_model=KBStats)
    def get_stats() -> KBStats:
        """获取知识库统计信息。"""
        from sqlalchemy import text as sql_text

        stats = KBStats(backend=cfg.backend, embedding_model=cfg.embeddings.model)

        with registry._engine.connect() as conn:
            doc_count = conn.execute(
                sql_text("SELECT COUNT(*) FROM documents WHERE status = 'active'")
            ).scalar()
            total_docs = conn.execute(
                sql_text("SELECT COUNT(*) FROM documents")
            ).scalar()
            chunk_count = conn.execute(
                sql_text("SELECT COUNT(*) FROM chunks")
            ).scalar()
            last_run = conn.execute(
                sql_text(
                    "SELECT finished_at FROM ingestion_runs WHERE status = 'completed' ORDER BY finished_at DESC LIMIT 1"
                )
            ).scalar()

        stats.total_documents = total_docs or 0
        stats.active_documents = doc_count or 0
        stats.total_chunks = chunk_count or 0
        if last_run:
            stats.last_ingestion = str(last_run)

        return stats

    @router.get("/documents", response_model=List[DocumentInfo])
    def list_documents() -> List[DocumentInfo]:
        """列出所有已索引文档。"""
        from sqlalchemy import text as sql_text

        docs: List[DocumentInfo] = []
        with registry._engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT document_id, source_file, title, doc_type, status, size, created_at "
                    "FROM documents ORDER BY created_at DESC LIMIT 200"
                )
            ).fetchall()

        for row in rows:
            docs.append(
                DocumentInfo(
                    document_id=row[0],
                    source_file=row[1],
                    title=row[2],
                    doc_type=row[3],
                    status=row[4],
                    size=row[5] or 0,
                    created_at=str(row[6]) if row[6] else "",
                )
            )

        return docs

    @router.post("/documents/upload")
    async def upload_documents(files: List[UploadFile] = File(...)):
        """上传文档到知识库目录。"""
        saved = []
        errors = []

        for file in files:
            if not file.filename:
                continue

            filename = Path(file.filename).name
            ext = Path(filename).suffix.lower()

            if ext == ".pdf":
                dest = _raw_pdf_dir() / filename
            else:
                dest = _raw_docs_dir() / filename

            # Avoid overwriting: append suffix if exists
            if dest.exists():
                stem = dest.stem
                dest = dest.with_name(f"{stem}_{uuid.uuid4().hex[:6]}{ext}")

            try:
                content = await file.read()
                dest.write_bytes(content)
                saved.append(str(dest.name))
            except Exception as exc:
                errors.append(f"{filename}: {exc}")

        return {
            "saved": saved,
            "errors": errors,
            "message": f"成功上传 {len(saved)} 个文件"
            + (f"，{len(errors)} 个失败" if errors else ""),
        }

    @router.delete("/documents/{document_id}")
    def delete_document(document_id: str):
        """删除文档及其所有分块。"""
        doc = registry.get_document(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="文档不存在。")

        # Delete chunks from registry
        registry.delete_chunks_by_document_id(document_id)
        registry.delete_by_document_id(document_id)

        # Try to delete from vector stores
        try:
            if cfg.backend in ("chroma", "both"):
                from coal_kb.infra.persistence.vector import ChromaStore
                from coal_kb.infra.providers.embeddings import EmbeddingsConfig

                chroma = ChromaStore(
                    persist_dir=cfg.paths.chroma_dir,
                    collection_name=cfg.chroma.collection_name,
                    embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
                    embedding_model=cfg.embedding.model_name,
                )
                chroma.delete_where({"document_id": document_id})
        except Exception:
            pass

        try:
            if cfg.backend in ("elastic", "both"):
                from coal_kb.infra.persistence.search import ElasticStore

                es = ElasticStore(
                    host=cfg.elastic.host,
                    verify_certs=cfg.elastic.verify_certs,
                    timeout_s=cfg.elastic.timeout_s,
                )
                es.delete_by_document_id(cfg.elastic.alias_current, document_id)
        except Exception:
            pass

        return {"deleted": True, "document_id": document_id}

    @router.post("/ingest", response_model=IngestResult)
    def run_ingestion(rebuild: bool = False, force: bool = False) -> IngestResult:
        """触发文档摄入流程。"""
        try:
            pipeline = IngestPipeline(cfg=cfg)
            result = pipeline.run(rebuild=rebuild, force=force)
            return IngestResult(
                status="completed",
                message=f"摄入完成: {result.get('documents', 0)} 文档, {result.get('chunks', 0)} 分块",
                stats=result,
            )
        except Exception as exc:
            return IngestResult(
                status="failed",
                message=f"摄入失败: {exc}",
            )

    return router
