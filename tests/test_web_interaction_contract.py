"""验证 Provider、对话和上传状态的前端交互契约。"""

from pathlib import Path


STATIC_DIR = Path("src/coal_kb/interfaces/web/static")


def test_web_ui_exposes_friendly_provider_conversation_and_upload_states() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    style = (STATIC_DIR / "usability.css").read_text(encoding="utf-8")

    for marker in (
        'id="provider-effective-summary"',
        'id="provider-status-tokenizer"',
        'id="provider-status-embeddings"',
        'id="provider-status-rerank"',
        'id="provider-status-llm"',
        'id="conversation-status"',
        'id="upload-summary"',
        'id="setting-research-route"',
        'aria-live="polite"',
    ):
        assert marker in html

    for marker in (
        "api_key_configured",
        "data-retry-id",
        "local_state",
        "FILE_STATE_LABELS",
        "research_route",
        "applyTaskToFiles",
        "setButtonBusy",
    ):
        assert marker in script

    for marker in (
        ".provider-status.ready",
        ".delivery-state.failed",
        ".file-state.completed",
        ".toast-region",
    ):
        assert marker in style


def test_web_ui_has_no_duplicate_research_route_script() -> None:
    assert not (STATIC_DIR / "research_routes.js").exists()
