from __future__ import annotations

from coal_kb.api.models import RuntimeSettingsRequest, SettingsDefaultsResponse
from coal_kb.settings import AppConfig


def apply_runtime_overrides(cfg: AppConfig, payload: RuntimeSettingsRequest) -> AppConfig:
    runtime_cfg = cfg.model_copy(deep=True)

    if payload.provider_base_url:
        provider_base_url = payload.provider_base_url.strip()
        runtime_cfg.llm.base_url = provider_base_url
        runtime_cfg.embeddings.base_url = provider_base_url

    if payload.api_key:
        api_key = payload.api_key.strip()
        runtime_cfg.llm.api_key = api_key
        runtime_cfg.embeddings.api_key = api_key

    if payload.llm_model:
        runtime_cfg.llm.model = payload.llm_model.strip()

    if payload.embedding_model:
        runtime_cfg.embeddings.model = payload.embedding_model.strip()

    return runtime_cfg


def build_settings_defaults(cfg: AppConfig) -> SettingsDefaultsResponse:
    return SettingsDefaultsResponse(
        api_base_url="",
        provider_base_url=cfg.llm.base_url,
        llm_provider=cfg.llm.provider,
        llm_model=cfg.llm.model,
        embedding_model=cfg.embeddings.model,
        backend=cfg.backend,
        mode=cfg.retrieval.mode,
        k=cfg.retrieval.k,
        rerank=cfg.retrieval.rerank_enabled,
        llm=False,
        debug=False,
        backend_options=["elastic", "chroma", "both"],
        mode_options=["strict", "balanced", "broad"],
        llm_provider_options=["none", "dashscope", "openai_compatible", "openai"],
        notes=[
            "Embedding model overrides should match the embedding space used when the index was built.",
            "Provider base URL and API key are applied at request time and are not persisted on the server.",
        ],
    )
