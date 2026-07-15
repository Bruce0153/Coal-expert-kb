"""定义远程 API 与本地部署严格隔离的 Provider 配置。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderMode = Literal["remote", "local"]


class RemoteProviderConfig(BaseModel):
    """远程 API 配置，密钥只存在于该配置中。"""

    provider: str = "openai_compatible"
    base_url: str
    api_key_env: str
    api_key: str | None = None
    model: str
    timeout: int = 60
    endpoint: str | None = None
    dimensions: int | None = None


class LocalProviderConfig(BaseModel):
    """本地模型或本地兼容服务配置，不接受远程 API Key。"""

    provider: str = "huggingface"
    model: str
    model_path: str | None = None
    base_url: str | None = None
    device: str = "auto"
    timeout: int = 60
    dimensions: int | None = None
    trust_remote_code: bool = False


class CapabilityProviderConfig(BaseModel):
    """为单项能力选择远程或本地实现。"""

    mode: ProviderMode
    remote: RemoteProviderConfig
    local: LocalProviderConfig

    @property
    def active(self) -> RemoteProviderConfig | LocalProviderConfig:
        return self.remote if self.mode == "remote" else self.local

    @property
    def provider(self) -> str:
        return self.active.provider

    @property
    def model(self) -> str:
        return self.active.model

    @property
    def base_url(self) -> str:
        return self.active.base_url or ""

    @property
    def timeout(self) -> int:
        return self.active.timeout

    @property
    def dimensions(self) -> int | None:
        return self.active.dimensions


class TokenizerProviderConfig(CapabilityProviderConfig):
    """Tokenizer Provider 配置。"""


class EmbeddingsProviderConfig(CapabilityProviderConfig):
    """Embedding Provider 配置。"""


class RerankProviderConfig(CapabilityProviderConfig):
    """Rerank Provider 配置。"""


class LLMProviderConfig(CapabilityProviderConfig):
    """LLM Provider 配置。"""

    temperature: float = Field(default=0.0, ge=0.0)


# 默认配置

def default_tokenizer_config() -> TokenizerProviderConfig:
    return TokenizerProviderConfig(
        mode="local",
        remote=RemoteProviderConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            api_key_env="TOKENIZER_API_KEY",
            model="remote-tokenizer",
            endpoint="/tokenize",
        ),
        local=LocalProviderConfig(provider="huggingface", model="Qwen/Qwen3-8B", trust_remote_code=True),
    )


def default_embeddings_config() -> EmbeddingsProviderConfig:
    return EmbeddingsProviderConfig(
        mode="remote",
        remote=RemoteProviderConfig(
            provider="dashscope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            model="text-embedding-v4",
            dimensions=1024,
        ),
        local=LocalProviderConfig(provider="huggingface", model="BAAI/bge-m3", dimensions=1024),
    )


def default_rerank_config() -> RerankProviderConfig:
    return RerankProviderConfig(
        mode="remote",
        remote=RemoteProviderConfig(
            provider="dashscope",
            base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank",
            api_key_env="DASHSCOPE_API_KEY",
            model="qwen3-rerank",
            endpoint="/text-rerank/text-rerank",
        ),
        local=LocalProviderConfig(provider="cross_encoder", model="BAAI/bge-reranker-base"),
    )


def default_llm_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        mode="remote",
        remote=RemoteProviderConfig(
            provider="dashscope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            model="qwen3.5-flash",
        ),
        local=LocalProviderConfig(
            provider="vllm",
            base_url="http://127.0.0.1:8001/v1",
            model="Qwen/Qwen3-8B",
        ),
    )
