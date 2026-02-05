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
    raw_docs_dir: str = "data/raw_docs"
    interim_dir: str = "data/interim"
    artifacts_dir: str = "data/artifacts"
    chroma_dir: str = "storage/chroma_db"
    sqlite_path: str = "storage/expert.db"  # records db
    manifest_path: str = "storage/manifest.json"


class LocalEmbeddingConfig(BaseModel):
    # Local embedding (fallback): e.g., HuggingFace bge-m3
    model_name: str = "BAAI/bge-m3"


class ChunkingProfile(BaseModel):
    chunk_size: int
    chunk_overlap: int


class ChunkingConfig(BaseModel):
    strategy: str = "markdown_hierarchical_semantic"
    max_parent_tokens: int = 1200
    max_child_tokens: int = 300
    overlap_tokens: int = 60
    similarity_threshold: float = 0.72
    heading_max_depth: int = 4
    embedding_backend: str = "local_st"  # local_st | existing_factory

    # legacy fallback options
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




class TwoStageRetrievalConfig(BaseModel):
    enabled: bool = True
    parent_k_candidates: int = 200
    parent_k_final: int = 60
    max_parents: int = 60
    child_k_candidates: int = 300
    child_k_final: int = 30
    allow_relax_in_stage2: bool = True


class PDFMarkdownConfig(BaseModel):
    enabled: bool = True
    heading_max_depth: int = 4
    two_column_mode: str = "auto"  # auto|on|off
    drop_headers_footers: bool = True
    min_heading_font_ratio: float = 1.15


class RetrievalConfig(BaseModel):
    # ✅ only k (no candidates)
    k: int = 5

    rrf_k: int = 60
    max_per_source: int = 2
    max_relax_steps: int = 2
    range_expand_schedule: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2])
    mode: str = "balanced"

    rerank_enabled: bool = True
    # local fallback (only used if API rerank not available)
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 10
    rerank_device: str = "auto"

    drop_sections: list[str] = Field(
        default_factory=lambda: ["references", "acknowledgements", "contents", "appendix"]
    )
    drop_reference_like: bool = True
    two_stage: TwoStageRetrievalConfig = Field(default_factory=TwoStageRetrievalConfig)


class RerankConfig(BaseModel):
    # Default: DashScope(OpenAI-compatible) qwen3-rerank
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "qwen3-rerank"
    timeout: int = 60


class LoggingConfig(BaseModel):
    level: str = "INFO"


class RegistryConfig(BaseModel):
    sqlite_path: str = "storage/kb.db"  # registry db (runs/query logs)


class ElasticConfig(BaseModel):
    host: str = "http://localhost:9200"
    index_prefix: str = "coal_kb_chunks"
    alias_current: str = "coal_kb_chunks_current"
    alias_prev: str = "coal_kb_chunks_prev"
    verify_certs: bool = False
    timeout_s: int = 60
    bulk_chunk_size: int = 200
    enable_icu_analyzer: bool = True


class IngestionConfig(BaseModel):
    drop_sections: list[str] = Field(
        default_factory=lambda: ["references", "acknowledgements", "contents", "appendix"]
    )
    drop_reference_like_unknown: bool = True
    include_exts: list[str] = Field(
        default_factory=lambda: [
            "pdf",
            "txt",
            "md",
            "html",
            "docx",
            "pptx",
            "csv",
            "xlsx",
            "json",
            "jsonl",
        ]
    )
    exclude_exts: list[str] = Field(default_factory=list)


class TenancyConfig(BaseModel):
    enabled: bool = False
    default_tenant_id: str = "default"
    enforce_tenant_filter: bool = True


class ModelVersionsConfig(BaseModel):
    # Used for index/version bookkeeping (especially with ES index.py build)
    embedding_version: str = "v1"


class QueryRewriteConfig(BaseModel):
    enable_llm: bool = False


class LLMConfig(BaseModel):
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "qwen-plus"
    temperature: float = 0.0
    timeout: int = 60


class RemoteEmbeddingsConfig(BaseModel):
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "text-embedding-v3"
    dimensions: Optional[int] = 1024


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)

    embedding: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    pdf_markdown: PDFMarkdownConfig = Field(default_factory=PDFMarkdownConfig)
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

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: RemoteEmbeddingsConfig = Field(default_factory=RemoteEmbeddingsConfig)


class EnvSettings(BaseSettings):
    """
    Environment variable overrides (keep minimal and stable).
    """
    model_config = SettingsConfigDict(env_prefix="COAL_KB_", extra="ignore")

    config: str = "configs/app.yaml"

    embed_model: Optional[str] = None
    chroma_dir: Optional[str] = None
    sqlite_path: Optional[str] = None
    log_level: Optional[str] = None

    llm_model: Optional[str] = None
    emb_model: Optional[str] = None


def _ensure_dirs(cfg: AppConfig) -> AppConfig:
    Path(cfg.paths.raw_pdfs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.raw_docs_dir).mkdir(parents=True, exist_ok=True)
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

    raw = _load_yaml_unique_keys(config_path)
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

    if env.llm_model:
        cfg.llm.model = env.llm_model
    if env.emb_model:
        cfg.embeddings.model = env.emb_model

    return _ensure_dirs(cfg)


def _load_yaml_unique_keys(path: Path) -> dict:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key in YAML: {key}")
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )

    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    except ValueError as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc
