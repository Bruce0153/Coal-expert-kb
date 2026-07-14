from coal_kb.infra.config import AppConfig
from coal_kb.interfaces.api.models import ChatRequest
from coal_kb.interfaces.api.runtime_overrides import (
    apply_runtime_overrides,
    build_settings_defaults,
)


def test_apply_runtime_overrides_updates_models_and_provider_settings():
    cfg = AppConfig()
    payload = ChatRequest(
        message="hello",
        llm=True,
        llm_provider="openai_compatible",
        api_key="test-key",
        provider_base_url="https://example.test/v1",
        llm_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
    )

    runtime_cfg = apply_runtime_overrides(cfg, payload)

    assert runtime_cfg.llm.api_key == "test-key"
    assert runtime_cfg.embeddings.api_key == "test-key"
    assert runtime_cfg.llm.base_url == "https://example.test/v1"
    assert runtime_cfg.embeddings.base_url == "https://example.test/v1"
    assert runtime_cfg.llm.model == "gpt-4.1-mini"
    assert runtime_cfg.embeddings.model == "text-embedding-3-small"


def test_build_settings_defaults_exposes_frontend_defaults():
    cfg = AppConfig()
    payload = build_settings_defaults(cfg)
    assert payload.backend == cfg.backend
    assert payload.mode == cfg.retrieval.mode
    assert "elastic" in payload.backend_options
    assert "balanced" in payload.mode_options
    assert "dashscope" in payload.llm_provider_options
