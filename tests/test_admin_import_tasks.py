"""验证运行中上传和增量摄入任务。"""

from __future__ import annotations

import time
from pathlib import Path

from coal_kb.application.admin import AdminService
from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.config import AppConfig


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.raw_pdfs_dir = str(tmp_path / "pdfs")
    cfg.paths.raw_docs_dir = str(tmp_path / "docs")
    cfg.paths.interim_dir = str(tmp_path / "interim")
    cfg.paths.chroma_dir = str(tmp_path / "chroma")
    cfg.paths.manifest_path = str(tmp_path / "manifest.json")
    cfg.paths.sqlite_path = str(tmp_path / "records.db")
    cfg.registry.sqlite_path = str(tmp_path / "registry.db")
    return cfg


def _wait(service: AdminService, task_id: str) -> dict:
    for _ in range(100):
        task = service.get_task(task_id)
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_upload_can_save_without_ingesting(tmp_path: Path) -> None:
    service = AdminService(RuntimeConfigStore(_config(tmp_path)))
    task = service.start_import([("paper.txt", b"coal gasification")], auto_ingest=False)
    completed = _wait(service, task["task_id"])
    assert completed["status"] == "completed"
    assert completed["stage"] == "saved"
    assert completed["progress"] == 100
    assert (tmp_path / "docs" / "paper.txt").read_bytes() == b"coal gasification"


def test_upload_can_trigger_incremental_ingestion(monkeypatch, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    observed: list[AppConfig] = []

    def fake_process(self, *, rebuild=False, force=False, elastic_index_override=None):
        observed.append(self.cfg)
        return {"docs_scanned": 1, "indexed": 3, "chunks": 3}

    monkeypatch.setattr("coal_kb.ingestion.pipeline.IngestPipeline.process", fake_process)
    service = AdminService(RuntimeConfigStore(cfg))
    task = service.start_import([("paper.md", b"# New knowledge")], auto_ingest=True)
    completed = _wait(service, task["task_id"])
    assert completed["status"] == "completed"
    assert completed["stage"] == "completed"
    assert completed["stats"]["indexed"] == 3
    assert observed[0].paths.raw_docs_dir == cfg.paths.raw_docs_dir


def test_completed_task_survives_service_restart(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    service = AdminService(RuntimeConfigStore(cfg))
    task = service.start_import([("paper.txt", b"persistent task")], auto_ingest=False)
    completed = _wait(service, task["task_id"])

    restarted = AdminService(RuntimeConfigStore(cfg))
    restored = restarted.get_task(task["task_id"])
    assert restored["status"] == "completed"
    assert restored["stage"] == "saved"
    assert restored["saved"] == completed["saved"]


def test_running_task_is_marked_interrupted_after_restart(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    service = AdminService(RuntimeConfigStore(cfg))
    task = service._new_task("test running task")
    service._update_task(task.task_id, status="running", stage="indexing", progress=45)

    restarted = AdminService(RuntimeConfigStore(cfg))
    restored = restarted.get_task(task.task_id)
    assert restored["status"] == "failed"
    assert restored["stage"] == "interrupted"
    assert restored["progress"] == 100
