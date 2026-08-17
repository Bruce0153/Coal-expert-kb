"""组装 FastAPI、应用用例和网页静态资源。"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore
from coal_kb.infra.config import load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.infra.security import (
    AdminAuth,
    PublicHeadersMiddleware,
    PublicRequestGuard,
    PublicSecurityPolicy,
    PublicSessionMiddleware,
)
from coal_kb.interfaces.api import config
from coal_kb.interfaces.api.routes_admin import build_admin_router
from coal_kb.interfaces.api.routes_ask import build_ask_router
from coal_kb.interfaces.api.routes_auth import build_auth_router
from coal_kb.interfaces.api.routes_chat import build_chat_router
from coal_kb.interfaces.api.routes_public import build_public_router
from coal_kb.interfaces.api.routes_settings import build_settings_router
from coal_kb.interfaces.web import web_static_dir
from coal_kb.operations import health_status, readiness_status

LOGGER = logging.getLogger("coal_kb.api")


def create_app() -> FastAPI:
    cfg = load_config()
    policy = PublicSecurityPolicy.from_env()
    setup_logging(cfg, logger_name="coal_kb.api")
    configs = RuntimeConfigStore(cfg)
    auth = AdminAuth(policy)
    guard = PublicRequestGuard(policy)
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description=config.API_DESCRIPTION,
    )
    app.add_middleware(PublicSessionMiddleware, policy=policy)
    app.add_middleware(PublicHeadersMiddleware, policy=policy)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(policy.allowed_origins) if policy.public_mode else config.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    conversations = ConversationService(ConversationStore(cfg.registry.sqlite_path))
    app.include_router(build_public_router(configs, policy))
    app.include_router(build_auth_router(auth))
    app.include_router(build_ask_router(configs, policy, guard))
    app.include_router(build_chat_router(configs, conversations, policy, guard))
    app.include_router(build_admin_router(configs, auth, policy))
    app.include_router(build_settings_router(configs, auth))

    static_dir = web_static_dir()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        if not policy.public_mode:
            raise exc
        request_id = str(getattr(request.state, "request_id", "unknown"))
        LOGGER.exception(
            "Unhandled request error request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "服务暂时不可用，请稍后再试。", "request_id": request_id},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return health_status()

    @app.get("/ready")
    def ready() -> JSONResponse:
        is_ready, payload = readiness_status(cfg)
        return JSONResponse(content=payload, status_code=200 if is_ready else 503)

    @app.get("/admin")
    def admin() -> FileResponse:
        return FileResponse(static_dir / "admin.html")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
