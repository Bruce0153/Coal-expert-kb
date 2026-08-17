"""持久化后台摄入任务元数据，避免服务重启后任务状态完全丢失。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ImportTaskStore:
    """将轻量任务状态写入现有 registry SQLite；不承担任务调度。"""

    def __init__(self, sqlite_path: str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_import_tasks (
                    task_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save(self, payload: dict[str, Any]) -> None:
        task_id = str(payload["task_id"])
        updated_at = str(payload.get("updated_at", ""))
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO admin_import_tasks(task_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (task_id, encoded, updated_at),
            )
            connection.commit()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM admin_import_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0]))

    def list_recent(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM admin_import_tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(json.loads(row[0])) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)
