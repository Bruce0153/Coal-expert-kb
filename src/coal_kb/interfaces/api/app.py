"""组装 FastAPI、应用用例和网页静态资源。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore
from coal_kb.infra.config import load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.interfaces.api import config
from coal_kb.interfaces.api.models import SettingsDefaultsResponse
from coal_kb.interfaces.api.routes_admin import build_admin_router
from coal_kb.interfaces.api.routes_ask import build_ask_router
from coal_kb.interfaces.api.routes_chat import build_chat_router
from coal_kb.interfaces.api.runtime_overrides import build_settings_defaults
from coal_kb.interfaces.web import web_static_dir
from coal_kb.operations import health_status


def create_app() -> FastAPI:
    cfg = load_config()
    setup_logging(cfg, logger_name="coal_kb.api")
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description=config.API_DESCRIPTION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ALLOWED_ORIGINS,
        allow_credentials=config.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    conversations = ConversationService(ConversationStore(cfg.registry.sqlite_path))
    app.include_router(build_ask_router(cfg))
    app.include_router(build_chat_router(cfg, conversations))
    app.include_router(build_admin_router(cfg))

    static_dir = web_static_dir()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return health_status()

    @app.get("/api/settings/defaults", response_model=SettingsDefaultsResponse)
    def settings_defaults() -> SettingsDefaultsResponse:
        return build_settings_defaults(cfg)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
