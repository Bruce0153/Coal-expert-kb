"""定义知识库管理、增量入库和任务状态 HTTP 路由。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from coal_kb.application.admin import AdminService
from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.security import AdminAuth, PublicSecurityPolicy


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


class ImportTaskResponse(BaseModel):
    task_id: str
    status: str
    stage: str
    message: str
    progress: int = Field(ge=0, le=100)
    saved: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stats: dict[str, Any] | None = None
    created_at: str
    updated_at: str


def build_admin_router(
    configs: RuntimeConfigStore,
    auth: AdminAuth | None = None,
    policy: PublicSecurityPolicy | None = None,
) -> APIRouter:
    policy = policy or PublicSecurityPolicy.from_env()
    auth = auth or AdminAuth(policy)
    router = APIRouter(
        prefix="/api/admin",
        tags=["admin"],
        dependencies=[Depends(auth.require_admin)],
    )
    service = AdminService(configs)

    @router.get("/stats", response_model=KBStats)
    def get_stats() -> KBStats:
        return KBStats.model_validate(service.get_stats())

    @router.get("/documents", response_model=list[DocumentInfo])
    def list_documents() -> list[DocumentInfo]:
        return [DocumentInfo.model_validate(item) for item in service.list_documents()]

    @router.post("/documents/upload", response_model=ImportTaskResponse)
    async def upload_documents(
        files: Annotated[list[UploadFile], File()],
        auto_ingest: bool = True,
    ) -> ImportTaskResponse:
        if not files:
            raise HTTPException(status_code=400, detail="至少需要选择一个文件。")
        if len(files) > policy.upload_max_files:
            raise HTTPException(status_code=413, detail=f"一次最多上传 {policy.upload_max_files} 个文件。")

        buffered: list[tuple[str, bytes]] = []
        total_size = 0
        allowed_exts = {item.lower() for item in policy.upload_allowed_exts}
        for upload in files:
            if not upload.filename:
                continue
            extension = Path(upload.filename).suffix.lower()
            if extension not in allowed_exts:
                raise HTTPException(status_code=415, detail=f"不支持的文件类型：{extension or '无扩展名'}。")
            content = await upload.read(policy.upload_max_file_bytes + 1)
            if len(content) > policy.upload_max_file_bytes:
                raise HTTPException(status_code=413, detail=f"文件过大：{upload.filename}。")
            total_size += len(content)
            if total_size > policy.upload_max_total_bytes:
                raise HTTPException(status_code=413, detail="本次上传文件总大小超过限制。")
            buffered.append((upload.filename, content))

        if not buffered:
            raise HTTPException(status_code=400, detail="至少需要选择一个文件。")
        return ImportTaskResponse.model_validate(
            service.start_import(buffered, auto_ingest=auto_ingest)
        )

    @router.get("/tasks/{task_id}", response_model=ImportTaskResponse)
    def get_task(task_id: str) -> ImportTaskResponse:
        try:
            task = service.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在。") from exc
        return ImportTaskResponse.model_validate(task)

    @router.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail="文档不存在。")
        return {"deleted": True, "document_id": document_id}

    @router.post("/ingest", response_model=ImportTaskResponse)
    def run_ingestion(rebuild: bool = False, force: bool = False) -> ImportTaskResponse:
        return ImportTaskResponse.model_validate(
            service.start_ingestion(rebuild=rebuild, force=force)
        )

    return router
