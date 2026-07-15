"""修正 infra config 的正式 Provider 配置导出。"""

from pathlib import Path

Path("src/coal_kb/infra/config/__init__.py").write_text(
    '''"""导出应用配置和 Provider 配置。"""

from coal_kb.infra.config.env import EnvSettings
from coal_kb.infra.config.loader import _ensure_dirs, _load_yaml_unique_keys, load_config
from coal_kb.infra.config.models import (
    AppConfig,
    ChromaConfig,
    ChunkingConfig,
    ChunkingProfile,
    ElasticConfig,
    IngestionConfig,
    LoggingConfig,
    ModelVersionsConfig,
    PathsConfig,
    PDFMarkdownConfig,
    QueryRewriteConfig,
    RegistryConfig,
    RetrievalConfig,
    TenancyConfig,
    TwoStageRetrievalConfig,
)
from coal_kb.infra.providers.config import (
    EmbeddingsProviderConfig,
    LLMProviderConfig,
    LocalProviderConfig,
    RemoteProviderConfig,
    RerankProviderConfig,
    TokenizerProviderConfig,
)

__all__ = [
    "AppConfig",
    "ChromaConfig",
    "ChunkingConfig",
    "ChunkingProfile",
    "ElasticConfig",
    "EmbeddingsProviderConfig",
    "EnvSettings",
    "IngestionConfig",
    "LLMProviderConfig",
    "LocalProviderConfig",
    "LoggingConfig",
    "ModelVersionsConfig",
    "PDFMarkdownConfig",
    "PathsConfig",
    "QueryRewriteConfig",
    "RegistryConfig",
    "RemoteProviderConfig",
    "RerankProviderConfig",
    "RetrievalConfig",
    "TenancyConfig",
    "TokenizerProviderConfig",
    "TwoStageRetrievalConfig",
    "_ensure_dirs",
    "_load_yaml_unique_keys",
    "load_config",
]
''',
    encoding="utf-8",
)
