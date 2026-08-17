"""定义管理员登录、退出和认证状态接口。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from coal_kb.infra.security import AdminAuth


class AdminLoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=512)


class AdminAuthStatus(BaseModel):
    authenticated: bool


def build_auth_router(auth: AdminAuth) -> APIRouter:
    router = APIRouter(prefix="/api/auth/admin", tags=["auth"])

    @router.get("/status", response_model=AdminAuthStatus)
    def auth_status(request: Request) -> AdminAuthStatus:
        return AdminAuthStatus(authenticated=auth.is_authenticated(request))

    @router.post("/login", response_model=AdminAuthStatus)
    def login(payload: AdminLoginRequest, response: Response) -> AdminAuthStatus:
        auth.login(payload.password, response)
        return AdminAuthStatus(authenticated=True)

    @router.post("/logout", response_model=AdminAuthStatus)
    def logout(response: Response) -> AdminAuthStatus:
        auth.logout(response)
        return AdminAuthStatus(authenticated=False)

    return router
