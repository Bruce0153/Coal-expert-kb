"""编排知识库上传、增量摄入、任务状态和文档管理。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from sqlalchemy import text as sql_text

from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.persistence.registry import RegistrySQLite
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.infra.security import build_upload_path


@dataclass
class ImportTask:
    task_id: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "任务已进入队列。"
    progress: int = 0
    saved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AdminService:
    """使用单工作线程串行执行增量摄入，避免同时写入索引。"""

    def __init__(self, configs: RuntimeConfigStore) -> None:
        self.configs = configs
        self._tasks: dict[str, ImportTask] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coal-kb-ingest")

    def get_stats(self) -> dict[str, Any]:
        cfg = self.configs.snapshot()
        registry = RegistrySQLite(cfg.registry.sqlite_path)
        with registry._engine.connect() as connection:
            active_documents = connection.execute(
                sql_text("SELECT COUNT(*) FROM documents WHERE status = 'active'")
            ).scalar()
            total_documents = connection.execute(sql_text("SELECT COUNT(*) FROM documents")).scalar()
            total_chunks = connection.execute(sql_text("SELECT COUNT(*) FROM chunks")).scalar()
            last_ingestion = connection.execute(
                sql_text(
                    "SELECT finished_at FROM ingestion_runs WHERE status IN ('completed', 'no_docs') "
                    "ORDER BY finished_at DESC LIMIT 1"
                )
            ).scalar()
        return {
            "total_documents": total_documents or 0,
            "active_documents": active_documents or 0,
            "total_chunks": total_chunks or 0,
            "last_ingestion": str(last_ingestion) if last_ingestion else None,
            "backend": cfg.backend,
            "embedding_model": cfg.embeddings.model,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        cfg = self.configs.snapshot()
        registry = RegistrySQLite(cfg.registry.sqlite_path)
        with registry._engine.connect() as connection:
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

    def start_import(self, files: list[tuple[str, bytes]], *, auto_ingest: bool = True) -> dict[str, Any]:
        task = self._new_task("等待保存上传文件。")
        self._executor.submit(self._process_import, task.task_id, files, auto_ingest)
        return self.get_task(task.task_id)

    def start_ingestion(self, *, rebuild: bool = False, force: bool = False) -> dict[str, Any]:
        task = self._new_task("等待执行知识库摄入。")
        self._executor.submit(self._process_ingestion, task.task_id, rebuild, force)
        return self.get_task(task.task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            return asdict(task)

    def delete_document(self, document_id: str) -> bool:
        cfg = self.configs.snapshot()
        registry = RegistrySQLite(cfg.registry.sqlite_path)
        if registry.get_document(document_id) is None:
            return False
        registry.delete_chunks_by_document_id(document_id)
        registry.delete_by_document_id(document_id)
        self._delete_from_chroma(cfg, document_id)
        self._delete_from_elasticsearch(cfg, document_id)
        return True

    def _new_task(self, message: str) -> ImportTask:
        task = ImportTask(task_id=uuid4().hex, message=message)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def _update_task(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            task = self._tasks[task_id]
            for key, value in changes.items():
                setattr(task, key, value)
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def _process_import(
        self,
        task_id: str,
        files: list[tuple[str, bytes]],
        auto_ingest: bool,
    ) -> None:
        saved: list[str] = []
        errors: list[str] = []
        self._update_task(
            task_id,
            status="running",
            stage="saving",
            message="正在校验并保存文件。",
            progress=10,
        )
        total = max(len(files), 1)
        for index, (filename, content) in enumerate(files, start=1):
            try:
                saved.append(self._save_uploaded_document(filename, content))
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
            self._update_task(
                task_id,
                saved=list(saved),
                errors=list(errors),
                progress=10 + int(index / total * 25),
                message=f"已处理 {index}/{len(files)} 个上传文件。",
            )
        if not saved:
            self._update_task(
                task_id,
                status="failed",
                stage="failed",
                message="没有文件保存成功。",
                progress=100,
            )
            return
        if not auto_ingest:
            self._update_task(
                task_id,
                status="completed",
                stage="saved",
                message=f"已保存 {len(saved)} 个文件，尚未写入检索索引。",
                progress=100,
            )
            return
        self._run_pipeline(task_id, rebuild=False, force=False)

    def _process_ingestion(self, task_id: str, rebuild: bool, force: bool) -> None:
        self._update_task(
            task_id,
            status="running",
            stage="preparing",
            message="正在准备知识库摄入。",
            progress=10,
        )
        self._run_pipeline(task_id, rebuild=rebuild, force=force)

    def _run_pipeline(self, task_id: str, *, rebuild: bool, force: bool) -> None:
        self._update_task(
            task_id,
            status="running",
            stage="indexing",
            message="正在解析、分块、向量化并更新检索索引。",
            progress=45,
        )
        try:
            from coal_kb.ingestion.pipeline import IngestPipeline

            result = IngestPipeline(cfg=self.configs.snapshot()).process(
                rebuild=rebuild,
                force=force,
            )
            self._update_task(
                task_id,
                status="completed",
                stage="completed",
                message=(
                    f"知识库已更新：扫描 {result.get('docs_scanned', 0)} 个文档，"
                    f"写入 {result.get('indexed', 0)} 个分块。"
                ),
                progress=100,
                stats=result,
            )
        except Exception as exc:
            self._update_task(
                task_id,
                status="failed",
                stage="failed",
                message=f"知识库更新失败：{exc}",
                progress=100,
            )

    def _save_uploaded_document(self, filename: str, content: bytes) -> str:
        cfg = self.configs.snapshot()
        safe_name = Path(filename).name
        extension = Path(safe_name).suffix.lower()
        directory = Path(cfg.paths.raw_pdfs_dir if extension == ".pdf" else cfg.paths.raw_docs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = build_upload_path(directory, safe_name)
        destination.write_bytes(content)
        return destination.name

    @staticmethod
    def _delete_from_chroma(cfg: Any, document_id: str) -> None:
        if cfg.backend not in {"chroma", "both"}:
            return
        store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=cfg.embeddings,
            embedding_model=cfg.embeddings.model,
        )
        store.delete_where({"document_id": document_id})

    @staticmethod
    def _delete_from_elasticsearch(cfg: Any, document_id: str) -> None:
        if cfg.backend not in {"elastic", "both"}:
            return
        store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        store.delete_by_document_id(cfg.elastic.alias_current, document_id)
