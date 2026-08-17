"""定义运行配置读取、更新和恢复接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.security import AdminAuth, PublicSecurityPolicy
from coal_kb.interfaces.api.models import RuntimeSettingsRequest, SettingsDefaultsResponse
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides, build_settings_defaults


def build_settings_router(configs: RuntimeConfigStore, auth: AdminAuth | None = None) -> APIRouter:
    auth = auth or AdminAuth(PublicSecurityPolicy.from_env())
    router = APIRouter(prefix="/api/settings", tags=["settings"])

    @router.get("/defaults", response_model=SettingsDefaultsResponse)
    def defaults() -> SettingsDefaultsResponse:
        return build_settings_defaults(configs.snapshot())

    @router.get("/runtime", response_model=SettingsDefaultsResponse)
    def runtime() -> SettingsDefaultsResponse:
        return build_settings_defaults(configs.snapshot())

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
