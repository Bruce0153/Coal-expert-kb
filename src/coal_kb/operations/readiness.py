"""生产就绪检查：本地持久化、检索后端与远程 Provider 配置。"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

from coal_kb.infra.config import AppConfig


def _remote_provider_ready(config: Any) -> bool:
    if config.mode != "remote":
        return True
    remote = config.remote
    return bool(remote.api_key or os.getenv(remote.api_key_env, "").strip())


def readiness_status(cfg: AppConfig) -> tuple[bool, dict[str, Any]]:
    """只执行轻量依赖检查，不发起收费模型请求。"""
    checks: dict[str, Any] = {}

    sqlite_path = Path(cfg.registry.sqlite_path)
    try:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            connection.execute("SELECT 1").fetchone()
        checks["sqlite"] = {"ok": os.access(sqlite_path.parent, os.W_OK)}
    except Exception as exc:
        checks["sqlite"] = {"ok": False, "error": type(exc).__name__}

    if cfg.backend in {"elastic", "both"}:
        client: Elasticsearch | None = None
        try:
            client = Elasticsearch(
                cfg.elastic.host,
                verify_certs=cfg.elastic.verify_certs,
                request_timeout=min(cfg.elastic.timeout_s, 5),
            )
            checks["elasticsearch"] = {"ok": bool(client.ping())}
        except Exception as exc:
            checks["elasticsearch"] = {"ok": False, "error": type(exc).__name__}
        finally:
            if client is not None:
                client.close()
    else:
        checks["elasticsearch"] = {"ok": True, "skipped": True}

    provider_checks = {
        "embeddings": _remote_provider_ready(cfg.embeddings),
        "llm": _remote_provider_ready(cfg.llm),
        "rerank": _remote_provider_ready(cfg.rerank) if cfg.retrieval.rerank_enabled else True,
        "tokenizer": _remote_provider_ready(cfg.tokenizer),
    }
    checks["providers"] = {
        "ok": all(provider_checks.values()),
        "items": provider_checks,
    }

    ready = all(bool(item.get("ok")) for item in checks.values())
    return ready, {"status": "ready" if ready else "not_ready", "checks": checks}
