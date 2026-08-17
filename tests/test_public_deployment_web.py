"""验证公网管理入口、安全响应头和正式部署契约。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/coal_kb/interfaces/web/static"


def test_admin_entry_uses_http_only_server_auth_flow() -> None:
    html = (STATIC / "admin.html").read_text(encoding="utf-8")
    script = (STATIC / "admin.js").read_text(encoding="utf-8")
    assert 'id="admin-login-form"' in html
    assert 'name="password"' in html
    assert "/api/auth/admin/status" in script
    assert "/api/auth/admin/login" in script
    assert "/api/auth/admin/logout" in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script


def test_public_security_headers_are_wired_into_api() -> None:
    headers = (ROOT / "src/coal_kb/infra/security/headers.py").read_text(encoding="utf-8")
    app = (ROOT / "src/coal_kb/interfaces/api/app.py").read_text(encoding="utf-8")
    for header in (
        "X-Request-ID",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
        "Content-Security-Policy",
    ):
        assert header in headers
    assert "PublicHeadersMiddleware" in app
    assert '@app.get("/admin")' in app


def test_official_deployment_contract_is_present() -> None:
    deployment = (ROOT / "docs/deployment.md").read_text(encoding="utf-8")
    railway = (ROOT / "railway.toml").read_text(encoding="utf-8")
    assert "/app/data" in deployment
    assert "elasticsearch.railway.internal:9200" in deployment
    assert "/ready" in deployment
    assert 'healthcheckPath = "/ready"' in railway
