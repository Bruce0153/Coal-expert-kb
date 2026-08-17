"""离线验证生产配置、持久化路径和 FastAPI 部署入口。"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    with TemporaryDirectory(prefix="coal-kb-production-") as data_root:
        os.environ.update(
            {
                "COAL_KB_CONFIG": "configs/prod.yaml",
                "COAL_KB_PUBLIC_MODE": "true",
                "COAL_KB_DATA_ROOT": data_root,
                "COAL_KB_ELASTIC_URL": "http://elasticsearch.railway.internal:9200",
                "COAL_KB_SESSION_SECRET": "session-secret-for-offline-production-check",
                "COAL_KB_ADMIN_SECRET": "admin-secret-for-offline-production-check",
                "COAL_KB_ALLOWED_ORIGINS": "https://example.test",
                "DASHSCOPE_API_KEY": "offline-placeholder",
            }
        )

        from coal_kb.infra.config import load_config
        from coal_kb.infra.providers.tokenizers import make_tokenizer
        from coal_kb.infra.security import PublicSecurityPolicy
        from coal_kb.interfaces.api.app import app

        cfg = load_config()
        policy = PublicSecurityPolicy.from_env()
        root = Path(data_root)

        assert cfg.backend == "elastic"
        assert cfg.elastic.host == "http://elasticsearch.railway.internal:9200"
        assert Path(cfg.registry.sqlite_path) == root / "kb.db"
        assert Path(cfg.paths.sqlite_path) == root / "expert.db"
        assert Path(cfg.paths.raw_pdfs_dir) == root / "raw_pdfs"
        assert Path(cfg.paths.chroma_dir) == root / "chroma_db"
        assert cfg.tokenizer.mode == "local"
        assert cfg.tokenizer.local.provider == "tiktoken"
        assert cfg.tokenizer.local.model == "cl100k_base"
        assert make_tokenizer(cfg.tokenizer).count_tokens("production smoke test") > 0
        assert policy.public_mode is True

        route_paths = {route.path for route in app.routes}
        for expected in (
            "/",
            "/admin",
            "/health",
            "/ready",
            "/api/chat",
            "/api/ask",
            "/api/auth/admin/login",
        ):
            assert expected in route_paths

        railway = tomllib.loads(Path("railway.toml").read_text(encoding="utf-8"))
        assert railway["build"]["builder"] == "DOCKERFILE"
        assert railway["deploy"]["healthcheckPath"] == "/ready"
        assert Path("Dockerfile").is_file()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/quality/check_production.py
