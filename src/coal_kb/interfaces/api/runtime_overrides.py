"""将请求级 Provider 设置覆盖到隔离配置副本。"""

from __future__ import annotations

from typing import Any

from coal_kb.infra.config import AppConfig
from coal_kb.interfaces.api.models import RuntimeSettingsRequest, SettingsDefaultsResponse
from coal_kb.research import ResearchRoute

REMOTE_PROVIDERS = [
    "openai",
    "openai_compatible",
    "dashscope",
    "deepseek",
    "siliconflow",
    "moonshot",
    "zhipu",
]
LOCAL_PROVIDERS = {
    "tokenizer": ["huggingface"],
    "embeddings": ["huggingface"],
    "rerank": ["cross_encoder"],
    "llm": ["openai_compatible", "vllm", "ollama"],
}


def _apply_capability(
    config: Any,
    *,
    mode: str | None,
    provider: str | None,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
) -> None:
    if mode:
        config.mode = mode
    active = config.remote if config.mode == "remote" else config.local
    if provider:
        active.provider = provider.strip()
    if base_url is not None:
        active.base_url = base_url.strip() or None
    if model:
        active.model = model.strip()
    if config.mode == "remote" and api_key is not None:
        config.remote.api_key = api_key.strip() or None


def apply_runtime_overrides(cfg: AppConfig, payload: RuntimeSettingsRequest) -> AppConfig:
    """分别覆盖四项能力，远程密钥只保留在进程内配置。"""
    runtime_cfg = cfg.model_copy(deep=True)
    _apply_capability(
        runtime_cfg.tokenizer,
        mode=payload.tokenizer_mode,
        provider=payload.tokenizer_provider,
        base_url=payload.tokenizer_base_url,
        api_key=payload.tokenizer_api_key,
        model=payload.tokenizer_model,
    )
    _apply_capability(
        runtime_cfg.embeddings,
        mode=payload.embedding_mode,
        provider=payload.embedding_provider,
        base_url=payload.embedding_base_url,
        api_key=payload.embedding_api_key,
        model=payload.embedding_model,
    )
    _apply_capability(
        runtime_cfg.rerank,
        mode=payload.rerank_mode,
        provider=payload.rerank_provider,
        base_url=payload.rerank_base_url,
        api_key=payload.rerank_api_key,
        model=payload.rerank_model,
    )
    _apply_capability(
        runtime_cfg.llm,
        mode=payload.llm_mode,
        provider=None if payload.llm_provider == "none" else payload.llm_provider,
        base_url=payload.llm_base_url,
        api_key=payload.llm_api_key,
        model=payload.llm_model,
    )
    if payload.backend:
        runtime_cfg.backend = payload.backend
    if payload.mode:
        runtime_cfg.retrieval.mode = payload.mode
    if payload.k is not None:
        runtime_cfg.retrieval.k = payload.k
    runtime_cfg.retrieval.rerank_enabled = payload.rerank
    return runtime_cfg


def _capability_defaults(config: Any) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "remote": config.remote.model_dump(exclude={"api_key"}),
        "local": config.local.model_dump(),
    }


def build_settings_defaults(cfg: AppConfig) -> SettingsDefaultsResponse:
    """向 UI 返回四项能力和研究路线选项。"""
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
        research_route=ResearchRoute.STANDARD.value,
        backend_options=["elastic", "chroma", "both"],
        mode_options=["strict", "balanced", "broad"],
        research_route_options=[route.value for route in ResearchRoute],
        provider_options={
            capability: {"remote": REMOTE_PROVIDERS, "local": providers}
            for capability, providers in LOCAL_PROVIDERS.items()
        },
        notes=[
            "设置会立即应用于后续问答和增量入库，但不会把 API Key 写入磁盘。",
            "Embedding 模型必须与现有索引向量空间一致；切换模型后应重建索引。",
            "Agent 路线只执行固定白名单动作并受最大步数约束。",
        ],
    )
