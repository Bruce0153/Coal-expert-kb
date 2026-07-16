"""验证 Provider、对话和增量上传的人机交互状态组件。"""

from pathlib import Path


STATIC = Path("src/coal_kb/interfaces/web/static")


def test_web_ui_exposes_runtime_status_components() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "conversation-context",
        "settings-apply-status",
        "provider-tokenizer-status",
        "provider-embeddings-status",
        "provider-rerank-status",
        "provider-llm-status",
        "upload-selection-summary",
        "upload-stage-list",
        "upload-task-details",
    ):
        assert f'id="{element_id}"' in index
    assert 'data-upload-stage="transfer"' in index
    assert 'data-upload-stage="indexing"' in index
    assert 'aria-live="polite"' in index


def test_web_ui_tracks_conversation_and_upload_failures() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "renderProviderStatus" in script
    assert "pendingMessage" in script
    assert "lastFailedMessage" in script
    assert "data-retry-message" in script
    assert "task.stage" in script
    assert "task.saved" in script
    assert "task.errors" in script
    assert "task.stats" in script
    assert "任务已进入单线程队列" in script


def test_web_ui_styles_status_and_stage_feedback() -> None:
    styles = (STATIC / "usability.css").read_text(encoding="utf-8")
    for selector in (
        ".provider-status",
        ".settings-apply-status",
        ".conversation-context",
        ".message-error",
        ".upload-stage.active",
        ".upload-stage.done",
        ".upload-stage.failed",
    ):
        assert selector in styles
