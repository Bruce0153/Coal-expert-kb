from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    raw_pdfs_dir: str = "data/raw_pdfs"
    interim_dir: str = "data/interim"
    artifacts_dir: str = "data/artifacts"
    chroma_dir: str = "storage/chroma_db"
    sqlite_path: str = "storage/expert.db"
    manifest_path: str = "storage/manifest.json"


# Local embedding (fallback): e.g., HuggingFace bge-m3
class LocalEmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"


class ChunkingProfile(BaseModel):
    chunk_size: int
    chunk_overlap: int


class ChunkingConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 120
    profile_by_section: dict[str, ChunkingProfile] = Field(
        default_factory=lambda: {
            "results": ChunkingProfile(chunk_size=900, chunk_overlap=150),
            "discussion": ChunkingProfile(chunk_size=900, chunk_overlap=150),
            "methods": ChunkingProfile(chunk_size=650, chunk_overlap=120),
            "conditions": ChunkingProfile(chunk_size=650, chunk_overlap=120),
            "unknown": ChunkingProfile(chunk_size=750, chunk_overlap=120),
        }
    )


class ChromaConfig(BaseModel):
    collection_name: str = "coal_gasification_papers"


class RetrievalConfig(BaseModel):
    k: int = 5
    candidates: int = 50
    max_per_source: int = 2
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 50
    rerank_device: str = "auto"
    drop_sections: list[str] = Field(
        default_factory=lambda: ["references", "acknowledgements", "contents", "appendix"]
    )
    drop_reference_like: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class RegistryConfig(BaseModel):
    sqlite_path: str = "storage/kb.db"


class ModelVersionsConfig(BaseModel):
    embedding_version: str = "v1"


class ElasticConfig(BaseModel):
    host: str = "http://localhost:9200"
    index_prefix: str = "coal_kb_chunks"
    alias_current: str = "coal_kb_chunks_current"
    alias_prev: str = "coal_kb_chunks_prev"
    verify_certs: bool = False
    timeout_s: int = 60
    bulk_chunk_size: int = 200


class IngestionConfig(BaseModel):
    drop_sections: list[str] = Field(
        default_factory=lambda: ["references", "acknowledgements", "contents", "appendix"]
    )
    drop_reference_like_unknown: bool = True


class QueryRewriteConfig(BaseModel):
    enable_llm: bool = False


class TenancyConfig(BaseModel):
    enabled: bool = False
    default_tenant_id: str = "default"
    enforce_tenant_filter: bool = True


# DashScope / OpenAI-compatible Chat LLM config
class LLMConfig(BaseModel):
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "qwen-plus"
    temperature: float = 0.0
    timeout: int = 60


# DashScope / OpenAI-compatible Embeddings config
class RemoteEmbeddingsConfig(BaseModel):
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "text-embedding-v4"
    dimensions: Optional[int] = 1024


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # Keep this for local fallback embeddings (HF/SentenceTransformers)
    embedding: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backend: str = "elastic"
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    model_versions: ModelVersionsConfig = Field(default_factory=ModelVersionsConfig)
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)

    # NEW: LLM + remote embeddings (DashScope/OpenAI-compatible)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: RemoteEmbeddingsConfig = Field(default_factory=RemoteEmbeddingsConfig)


class EnvSettings(BaseSettings):
    """
    Environment variable overrides (keep minimal and stable).
    """
    model_config = SettingsConfigDict(env_prefix="COAL_KB_", extra="ignore")

    config: str = "configs/app.yaml"

    # Existing overrides
    embed_model: Optional[str] = None
    chroma_dir: Optional[str] = None
    sqlite_path: Optional[str] = None
    log_level: Optional[str] = None

    # Optional overrides (useful but not required)
    llm_model: Optional[str] = None
    emb_model: Optional[str] = None


def _ensure_dirs(cfg: AppConfig) -> AppConfig:
    Path(cfg.paths.raw_pdfs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.interim_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.registry.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return cfg


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """
    Load YAML config + .env overrides.
    """
    load_dotenv(override=False)
    env = EnvSettings()

    config_path = Path(env.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Set COAL_KB_CONFIG or create configs/app.yaml."
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig.model_validate(raw)

    # Apply env overrides (keep minimal)
    if env.embed_model:
        cfg.embedding.model_name = env.embed_model
    if env.chroma_dir:
        cfg.paths.chroma_dir = env.chroma_dir
    if env.sqlite_path:
        cfg.paths.sqlite_path = env.sqlite_path
    if env.log_level:
        cfg.logging.level = env.log_level

    # Optional overrides
    if env.llm_model:
        cfg.llm.model = env.llm_model
    if env.emb_model:
        cfg.embeddings.model = env.emb_model

    return _ensure_dirs(cfg)
