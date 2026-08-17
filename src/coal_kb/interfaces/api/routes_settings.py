"""定义管理员运行配置与只读公网 bootstrap。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.security import AdminAuth, PublicSecurityPolicy
from coal_kb.interfaces.api.models import RuntimeSettingsRequest, SettingsDefaultsResponse
from coal_kb.interfaces.api.routes_public import build_public_config
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides, build_settings_defaults


def build_settings_router(configs: RuntimeConfigStore, auth: AdminAuth | None = None) -> APIRouter:
    policy = PublicSecurityPolicy.from_env()
    auth = auth or AdminAuth(policy)
    router = APIRouter(prefix="/api/settings", tags=["settings"])

    @router.get(
        "/defaults",
        response_model=SettingsDefaultsResponse,
        dependencies=[Depends(auth.require_admin)],
    )
    def defaults() -> SettingsDefaultsResponse:
        return build_settings_defaults(configs.snapshot())

    @router.get("/runtime", response_model=SettingsDefaultsResponse)
    def runtime(request: Request) -> SettingsDefaultsResponse:
        cfg = configs.snapshot()
        if auth.is_authenticated(request):
            return build_settings_defaults(cfg)
        return build_public_config(cfg, policy)

    @router.put(
        "/runtime",
        response_model=SettingsDefaultsResponse,
        dependencies=[Depends(auth.require_admin)],
    )
    def update_runtime(payload: RuntimeSettingsRequest) -> SettingsDefaultsResponse:
        updated = apply_runtime_overrides(configs.snapshot(), payload)
        configs.replace(updated)
        return build_settings_defaults(updated)

    @router.delete(
        "/runtime",
        response_model=SettingsDefaultsResponse,
        dependencies=[Depends(auth.require_admin)],
    )
    def reset_runtime() -> SettingsDefaultsResponse:
        return build_settings_defaults(configs.reset())

    return router
