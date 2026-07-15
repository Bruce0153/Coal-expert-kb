"""恢复 AppConfig 默认构造并修正 Provider 环境覆盖。"""

from pathlib import Path

provider_path = Path("src/coal_kb/infra/providers/config.py")
text = provider_path.read_text(encoding="utf-8")
text += '''

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
'''
provider_path.write_text(text, encoding="utf-8")

models_path = Path("src/coal_kb/infra/config/models.py")
models_text = models_path.read_text(encoding="utf-8")
models_text = models_text.replace(
    "from coal_kb.infra.providers.config import EmbeddingsProviderConfig, LLMProviderConfig, RerankProviderConfig, TokenizerProviderConfig",
    "from coal_kb.infra.providers.config import (\n    EmbeddingsProviderConfig,\n    LLMProviderConfig,\n    RerankProviderConfig,\n    TokenizerProviderConfig,\n    default_embeddings_config,\n    default_llm_config,\n    default_rerank_config,\n    default_tokenizer_config,\n)",
)
models_text = models_text.replace(
    "    tokenizer: TokenizerProviderConfig\n    embeddings: EmbeddingsProviderConfig\n    rerank: RerankProviderConfig\n    llm: LLMProviderConfig\n",
    "    tokenizer: TokenizerProviderConfig = Field(default_factory=default_tokenizer_config)\n    embeddings: EmbeddingsProviderConfig = Field(default_factory=default_embeddings_config)\n    rerank: RerankProviderConfig = Field(default_factory=default_rerank_config)\n    llm: LLMProviderConfig = Field(default_factory=default_llm_config)\n",
)
models_path.write_text(models_text, encoding="utf-8")

loader_path = Path("src/coal_kb/infra/config/loader.py")
loader_text = loader_path.read_text(encoding="utf-8")
loader_text = loader_text.replace("cfg.llm.model = env.llm_model", "cfg.llm.active.model = env.llm_model")
loader_text = loader_text.replace("cfg.embeddings.model = env.embeddings_model", "cfg.embeddings.active.model = env.embeddings_model")
loader_path.write_text(loader_text, encoding="utf-8")
