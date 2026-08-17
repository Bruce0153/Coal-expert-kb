"""签名匿名会话与管理员 Cookie。"""

from __future__ import annotations

import hashlib
import hmac
from uuid import uuid4

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

from .policy import PublicSecurityPolicy


def _sign(value: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{value}.{digest}"


def _verify(token: str | None, secret: str) -> str | None:
    if not token or "." not in token:
        return None
    value, digest = token.rsplit(".", 1)
    expected = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, expected):
        return None
    return value


class PublicSessionMiddleware(BaseHTTPMiddleware):
    """为每个浏览器分配签名匿名会话 ID，并写入 request.state。"""

    def __init__(self, app, policy: PublicSecurityPolicy) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.policy = policy

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        token = request.cookies.get(self.policy.session_cookie_name)
        session_id = _verify(token, self.policy.session_secret)
        created = session_id is None
        if session_id is None:
            session_id = uuid4().hex
        request.state.session_id = session_id
        response = await call_next(request)
        if created:
            response.set_cookie(
                key=self.policy.session_cookie_name,
                value=_sign(session_id, self.policy.session_secret),
                max_age=365 * 24 * 60 * 60,
                httponly=True,
                secure=self.policy.public_mode,
                samesite="lax",
                path="/",
            )
        return response


class AdminAuth:
    """无用户系统版本的管理员认证：一个服务端 Secret + 签名 Cookie。"""

    def __init__(self, policy: PublicSecurityPolicy) -> None:
        self.policy = policy

    def is_authenticated(self, request: Request) -> bool:
        if not self.policy.public_mode:
            return True
        token = request.cookies.get(self.policy.admin_cookie_name)
        return _verify(token, self.policy.admin_secret) == "admin"

    def require_admin(self, request: Request) -> None:
        if not self.is_authenticated(request):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="管理员认证失败。")

    def login(self, password: str, response: Response) -> None:
        if not hmac.compare_digest(password, self.policy.admin_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="管理员密码错误。")
        response.set_cookie(
            key=self.policy.admin_cookie_name,
            value=_sign("admin", self.policy.admin_secret),
            max_age=self.policy.admin_cookie_max_age,
            httponly=True,
            secure=self.policy.public_mode,
            samesite="strict",
            path="/",
        )

    def logout(self, response: Response) -> None:
        response.delete_cookie(self.policy.admin_cookie_name, path="/")
