"""将请求级 Provider 设置覆盖到配置副本。"""

from __future__ import annotations

from coal_kb.infra.config import AppConfig
from coal_kb.interfaces.api.models import RuntimeSettingsRequest, SettingsDefaultsResponse


def _apply_capability(config, *, mode, provider, base_url, api_key, model) -> None:
    if mode:
        config.mode = mode
    active = config.remote if config.mode == "remote" else config.local
    if provider:
        active.provider = provider.strip()
    if base_url:
        active.base_url = base_url.strip()
    if model:
        active.model = model.strip()
    if config.mode == "remote" and api_key:
        config.remote.api_key = api_key.strip()


def apply_runtime_overrides(cfg: AppConfig, payload: RuntimeSettingsRequest) -> AppConfig:
    """分别覆盖四项能力，远程密钥不会写入本地配置。"""
    runtime_cfg = cfg.model_copy(deep=True)
    _apply_capability(runtime_cfg.tokenizer, mode=payload.tokenizer_mode, provider=payload.tokenizer_provider, base_url=payload.tokenizer_base_url, api_key=payload.tokenizer_api_key, model=payload.tokenizer_model)
    _apply_capability(runtime_cfg.embeddings, mode=payload.embedding_mode, provider=payload.embedding_provider, base_url=payload.embedding_base_url, api_key=payload.embedding_api_key, model=payload.embedding_model)
    _apply_capability(runtime_cfg.rerank, mode=payload.rerank_mode, provider=payload.rerank_provider, base_url=payload.rerank_base_url, api_key=payload.rerank_api_key, model=payload.rerank_model)
    _apply_capability(runtime_cfg.llm, mode=payload.llm_mode, provider=None if payload.llm_provider == "none" else payload.llm_provider, base_url=payload.llm_base_url, api_key=payload.llm_api_key, model=payload.llm_model)
    return runtime_cfg


def _capability_defaults(config) -> dict:
    return {
        "mode": config.mode,
        "remote": config.remote.model_dump(exclude={"api_key"}),
        "local": config.local.model_dump(),
    }


def build_settings_defaults(cfg: AppConfig) -> SettingsDefaultsResponse:
    """向 UI 返回四项能力的独立配置。"""
    return SettingsDefaultsResponse(
        api_base_url="",
        tokenizer=_capability_defaults(cfg.tokenizer),
        embeddings=_capability_defaults(cfg.embeddings),
        rerank_config=_capability_defaults(cfg.rerank),
        llm_config=_capability_defaults(cfg.llm),
        backend=cfg.backend,
        mode=cfg.retrieval.mode,
        k=cfg.retrieval.k,
        rerank=cfg.retrieval.rerank_enabled,
        llm=True,
        debug=False,
        backend_options=["elastic", "chroma", "both"],
        mode_options=["strict", "balanced", "broad"],
        remote_provider_options=["openai", "openai_compatible", "dashscope", "deepseek", "siliconflow", "moonshot", "zhipu"],
        local_provider_options={
            "tokenizer": ["huggingface"],
            "embeddings": ["huggingface"],
            "rerank": ["cross_encoder"],
            "llm": ["openai_compatible", "vllm", "ollama"],
        },
        notes=[
            "远程模式需要独立 API Key；本地模式不会读取或持久化远程密钥。",
            "Embedding 模型必须与索引构建时的向量空间一致。",
            "Reranker 已正式接入检索链，Tokenizer 用于真实上下文 Token 预算。",
        ],
    )
