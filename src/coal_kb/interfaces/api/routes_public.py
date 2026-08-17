"""定义普通访客可读取的最小运行配置。"""

from __future__ import annotations

from fastapi import APIRouter

from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.security import PublicSecurityPolicy
from coal_kb.interfaces.api.models import SettingsDefaultsResponse
from coal_kb.interfaces.api.runtime_overrides import build_settings_defaults


def build_public_router(
    configs: RuntimeConfigStore,
    policy: PublicSecurityPolicy,
) -> APIRouter:
    router = APIRouter(prefix="/api/public", tags=["public"])

    @router.get("/config", response_model=SettingsDefaultsResponse)
    def public_config() -> SettingsDefaultsResponse:
        response = build_settings_defaults(configs.snapshot())
        if not policy.public_mode:
            return response
        response.backend_options = [response.backend]
        response.mode_options = [response.mode]
        response.research_route_options = list(policy.allowed_research_routes)
        response.debug = False
        response.provider_options = {}
        response.notes = ["公网访客仅使用服务器已配置的模型与检索参数。"]
        return response

    return router
