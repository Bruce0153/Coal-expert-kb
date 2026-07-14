"""验证 application 与 interfaces 的单向依赖边界。"""

from pathlib import Path

from coal_kb.application.ask import AskExecution, build_response_payload
from coal_kb.application.chat import ChatOrchestrator
from coal_kb.interfaces.api.app import create_app
from coal_kb.interfaces.api.models import ChatRequest
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides
from coal_kb.interfaces.web import web_static_dir

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def test_canonical_application_and_interface_exports() -> None:
    assert AskExecution.__module__ == "coal_kb.application.ask"
    assert build_response_payload.__module__ == "coal_kb.application.ask"
    assert ChatOrchestrator.__module__ == "coal_kb.application.chat"
    assert ChatRequest.__module__ == "coal_kb.interfaces.api.models"
    assert apply_runtime_overrides.__module__ == "coal_kb.interfaces.api.runtime_overrides"


def test_application_layer_does_not_depend_on_fastapi_or_web() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "application").glob("*.py"))
    assert "fastapi" not in text
    assert "coal_kb.interfaces" not in text
    assert "coal_kb.web" not in text


def test_interface_app_preserves_public_routes_and_static_path() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    assert {"/", "/health", "/api/ask", "/api/chat", "/api/conversations", "/api/admin/stats"} <= paths
    static_dir = web_static_dir()
    assert (static_dir / "index.html").exists()
    assert (static_dir / "app.js").exists()


def test_removed_transport_packages_are_absent() -> None:
    assert not (ROOT / "api").exists()
    assert not (ROOT / "chat").exists()
    assert not (ROOT / "web").exists()
