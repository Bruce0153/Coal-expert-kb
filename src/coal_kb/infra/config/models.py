from __future__ import annotations

from pydantic import BaseModel, Field

from coal_kb.infra.providers.config import (
    EmbeddingsProviderConfig,
    LLMProviderConfig,
    RerankProviderConfig,
    TokenizerProviderConfig,
    default_embeddings_config,
    default_llm_config,
    default_rerank_config,
    default_tokenizer_config,
)


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

    # section-aware fallback options
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



class ComplexQAConfig(BaseModel):
    """定义 Milestone C 路由、数据源和上下文预算。"""

    enabled: bool = True
    max_subqueries: int = 4
    max_multi_hop_steps: int = 3
    comparison_k_per_side: int = 4
    cross_document_min_sources: int = 2
    cross_document_max_per_source: int = 2
    aggregation_record_limit: int = 500
    aggregation_evidence_limit: int = 12
    table_records_path: str = "data/interim/table_records.jsonl"
    table_top_k: int = 5
    base_context_tokens: int = 2400
    base_evidence_chunks: int = 10


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




class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)

    tokenizer: TokenizerProviderConfig = Field(default_factory=default_tokenizer_config)
    embeddings: EmbeddingsProviderConfig = Field(default_factory=default_embeddings_config)
    rerank: RerankProviderConfig = Field(default_factory=default_rerank_config)
    llm: LLMProviderConfig = Field(default_factory=default_llm_config)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    pdf_markdown: PDFMarkdownConfig = Field(default_factory=PDFMarkdownConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    complex_qa: ComplexQAConfig = Field(default_factory=ComplexQAConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    backend: str = "elastic"
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    model_versions: ModelVersionsConfig = Field(default_factory=ModelVersionsConfig)
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)

