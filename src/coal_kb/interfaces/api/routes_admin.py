"""定义知识库管理 HTTP 路由。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from coal_kb.application.admin import AdminService
from coal_kb.infra.config import AppConfig


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
    stats: dict[str, Any] | None = None


def build_admin_router(cfg: AppConfig) -> APIRouter:
    router = APIRouter(prefix="/api/admin", tags=["admin"])
    service = AdminService(cfg)

    @router.get("/stats", response_model=KBStats)
    def get_stats() -> KBStats:
        return KBStats.model_validate(service.get_stats())

    @router.get("/documents", response_model=list[DocumentInfo])
    def list_documents() -> list[DocumentInfo]:
        return [DocumentInfo.model_validate(item) for item in service.list_documents()]

    @router.post("/documents/upload")
    async def upload_documents(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
        saved: list[str] = []
        errors: list[str] = []
        for upload in files:
            if not upload.filename:
                continue
            try:
                saved.append(service.save_uploaded_document(upload.filename, await upload.read()))
            except Exception as exc:
                errors.append(f"{upload.filename}: {exc}")
        return {
            "saved": saved,
            "errors": errors,
            "message": f"成功上传 {len(saved)} 个文件" + (f"，{len(errors)} 个失败" if errors else ""),
        }

    @router.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail="文档不存在。")
        return {"deleted": True, "document_id": document_id}

    @router.post("/ingest", response_model=IngestResult)
    def run_ingestion(rebuild: bool = False, force: bool = False) -> IngestResult:
        return IngestResult.model_validate(service.run_ingestion(rebuild=rebuild, force=force))

    return router
