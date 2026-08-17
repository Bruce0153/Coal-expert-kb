"""公网部署安全策略与环境变量解析。"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


@dataclass(frozen=True)
class PublicSecurityPolicy:
    """只描述公网访问边界，不承载模型或检索业务配置。"""

    public_mode: bool
    session_secret: str
    admin_secret: str
    allowed_origins: tuple[str, ...]
    allowed_research_routes: tuple[str, ...]
    max_query_chars: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    max_concurrent_queries: int
    session_cookie_name: str
    admin_cookie_name: str
    admin_cookie_max_age: int
    upload_max_files: int
    upload_max_file_bytes: int
    upload_max_total_bytes: int
    upload_allowed_exts: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "PublicSecurityPolicy":
        public_mode = _env_bool("COAL_KB_PUBLIC_MODE", False)
        session_secret = os.getenv("COAL_KB_SESSION_SECRET", "")
        admin_secret = os.getenv("COAL_KB_ADMIN_SECRET", "")
        if public_mode:
            if len(session_secret) < 24:
                raise RuntimeError("COAL_KB_SESSION_SECRET must contain at least 24 characters in public mode")
            if len(admin_secret) < 12:
                raise RuntimeError("COAL_KB_ADMIN_SECRET must contain at least 12 characters in public mode")
        else:
            session_secret = session_secret or "coal-kb-development-session-secret"
            admin_secret = admin_secret or "coal-kb-development-admin-secret"

        return cls(
            public_mode=public_mode,
            session_secret=session_secret,
            admin_secret=admin_secret,
            allowed_origins=_env_csv(
                "COAL_KB_ALLOWED_ORIGINS",
                ("http://127.0.0.1:8000", "http://localhost:8000"),
            ),
            allowed_research_routes=_env_csv(
                "COAL_KB_PUBLIC_RESEARCH_ROUTES",
                ("standard", "graph"),
            ),
            max_query_chars=_env_int("COAL_KB_MAX_QUERY_CHARS", 4000, minimum=128),
            rate_limit_requests=_env_int("COAL_KB_RATE_LIMIT_REQUESTS", 30),
            rate_limit_window_seconds=_env_int("COAL_KB_RATE_LIMIT_WINDOW_SECONDS", 60),
            max_concurrent_queries=_env_int("COAL_KB_MAX_CONCURRENT_QUERIES", 4),
            session_cookie_name=os.getenv("COAL_KB_SESSION_COOKIE_NAME", "coal_kb_session"),
            admin_cookie_name=os.getenv("COAL_KB_ADMIN_COOKIE_NAME", "coal_kb_admin"),
            admin_cookie_max_age=_env_int("COAL_KB_ADMIN_COOKIE_MAX_AGE", 12 * 60 * 60),
            upload_max_files=_env_int("COAL_KB_UPLOAD_MAX_FILES", 10),
            upload_max_file_bytes=_env_int("COAL_KB_UPLOAD_MAX_FILE_BYTES", 25 * 1024 * 1024),
            upload_max_total_bytes=_env_int("COAL_KB_UPLOAD_MAX_TOTAL_BYTES", 100 * 1024 * 1024),
            upload_allowed_exts=_env_csv(
                "COAL_KB_UPLOAD_ALLOWED_EXTS",
                (".pdf", ".txt", ".md", ".html", ".docx", ".pptx", ".csv", ".xlsx", ".json", ".jsonl"),
            ),
        )
