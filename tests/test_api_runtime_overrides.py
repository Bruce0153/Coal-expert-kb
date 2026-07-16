from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.config import AppConfig
from coal_kb.interfaces.api.models import ChatRequest
from coal_kb.interfaces.api.runtime_overrides import (
    apply_runtime_overrides,
    build_settings_defaults,
)


def test_apply_runtime_overrides_updates_each_capability() -> None:
    cfg = AppConfig()
    payload = ChatRequest(
        message="hello",
        llm=True,
        backend="chroma",
        mode="broad",
        k=9,
        tokenizer_mode="remote",
        tokenizer_provider="openai_compatible",
        tokenizer_base_url="https://tokenizer.example/v1",
        tokenizer_api_key="tokenizer-key",
        tokenizer_model="tokenizer-model",
        embedding_mode="remote",
        embedding_provider="dashscope",
        embedding_base_url="https://embedding.example/v1",
        embedding_api_key="embedding-key",
        embedding_model="embedding-model",
        rerank_mode="local",
        rerank_provider="cross_encoder",
        rerank_model="rerank-model",
        llm_mode="remote",
        llm_provider="openai_compatible",
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-key",
        llm_model="llm-model",
    )

    runtime_cfg = apply_runtime_overrides(cfg, payload)

    assert runtime_cfg.backend == "chroma"
    assert runtime_cfg.retrieval.mode == "broad"
    assert runtime_cfg.retrieval.k == 9
    assert runtime_cfg.tokenizer.remote.api_key == "tokenizer-key"
    assert runtime_cfg.embeddings.remote.model == "embedding-model"
    assert runtime_cfg.rerank.local.model == "rerank-model"
    assert runtime_cfg.llm.remote.base_url == "https://llm.example/v1"
    assert runtime_cfg.llm.remote.api_key == "llm-key"


def test_runtime_config_store_returns_isolated_snapshots() -> None:
    store = RuntimeConfigStore(AppConfig())
    first = store.snapshot()
    first.backend = "chroma"
    assert store.snapshot().backend != "chroma"
    store.replace(first)
    assert store.snapshot().backend == "chroma"
    assert store.reset().backend == AppConfig().backend


def test_build_settings_defaults_exposes_provider_options_without_keys() -> None:
    cfg = AppConfig()
    payload = build_settings_defaults(cfg)
    assert payload.backend == cfg.backend
    assert "elastic" in payload.backend_options
    assert "dashscope" in payload.provider_options["llm"]["remote"]
    assert "cross_encoder" in payload.provider_options["rerank"]["local"]
    assert "api_key" not in payload.llm_config["remote"]
