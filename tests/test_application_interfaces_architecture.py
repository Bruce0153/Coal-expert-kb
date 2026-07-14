"""验证 application 与 interfaces 分层和旧入口兼容。"""

from pathlib import Path

from coal_kb.api.models import ChatRequest as LegacyChatRequest
from coal_kb.api.runtime_overrides import apply_runtime_overrides as legacy_apply_overrides
from coal_kb.application.ask import AskExecution as CanonicalAskExecution
from coal_kb.application.ask import build_response_payload as canonical_build_response_payload
from coal_kb.application.chat import ChatOrchestrator as CanonicalChatOrchestrator
from coal_kb.chat.orchestrator import ChatOrchestrator as LegacyChatOrchestrator
from coal_kb.interfaces.api.app import create_app
from coal_kb.interfaces.api.models import ChatRequest as CanonicalChatRequest
from coal_kb.interfaces.api.runtime_overrides import (
    apply_runtime_overrides as canonical_apply_overrides,
)
from coal_kb.interfaces.web import web_static_dir
from coal_kb.qa.ask_pipeline import AskExecution as LegacyAskExecution
from coal_kb.qa.ask_pipeline import build_response_payload as legacy_build_response_payload

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def test_legacy_application_exports_remain_compatible() -> None:
    assert LegacyAskExecution is CanonicalAskExecution
    assert legacy_build_response_payload is canonical_build_response_payload
    assert issubclass(LegacyChatOrchestrator, CanonicalChatOrchestrator)
    assert LegacyChatRequest is CanonicalChatRequest
    assert legacy_apply_overrides is canonical_apply_overrides


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
