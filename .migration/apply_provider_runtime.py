"""构建远程与本地隔离的多供应商模型运行时。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Missing replacement target in {path}: {old[:80]}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def process() -> None:
    _write(
        "src/coal_kb/infra/providers/config.py",
        '''"""定义远程 API 与本地部署严格隔离的 Provider 配置。"""

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
''',
    )
    _write(
        "src/coal_kb/infra/providers/utils/http.py",
        '''"""提供远程 Provider 共用的密钥解析和 HTTP 响应校验。"""

from __future__ import annotations

import os
from typing import Any


def resolve_remote_api_key(*, api_key: str | None, api_key_env: str) -> str:
    """只为远程 Provider 解析 API Key。"""
    resolved = (api_key or os.getenv(api_key_env) or "").strip()
    if not resolved:
        raise RuntimeError(f"Missing remote API key: {api_key_env}")
    return resolved


def extract_json_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    """从常见远程响应字段中提取列表。"""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []
''',
    )
    _write(
        "src/coal_kb/infra/providers/embeddings/remote/openai_compatible.py",
        '''"""通过 OpenAI 兼容远程 API 创建 Embedding。"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key

SUPPORTED_REMOTE_EMBEDDING_PROVIDERS = {
    "openai",
    "openai_compatible",
    "dashscope",
    "deepseek",
    "siliconflow",
    "moonshot",
    "zhipu",
}


def make_remote_embeddings(config: RemoteProviderConfig) -> OpenAIEmbeddings:
    """创建需要 API Key 的远程 Embedding 客户端。"""
    if config.provider not in SUPPORTED_REMOTE_EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported remote embedding provider: {config.provider}")
    kwargs: dict[str, object] = {}
    if config.dimensions is not None:
        kwargs["dimensions"] = config.dimensions
    if config.provider == "dashscope":
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 10
    return OpenAIEmbeddings(
        model=config.model,
        api_key=resolve_remote_api_key(api_key=config.api_key, api_key_env=config.api_key_env),
        base_url=config.base_url,
        **kwargs,
    )
''',
    )
    _write(
        "src/coal_kb/infra/providers/embeddings/local/huggingface.py",
        '''"""从本地 HuggingFace 模型目录创建 Embedding。"""

from __future__ import annotations

from coal_kb.infra.providers.config import LocalProviderConfig


def make_local_embeddings(config: LocalProviderConfig):
    """创建不读取远程 API Key 的本地 Embedding。"""
    if config.provider != "huggingface":
        raise ValueError(f"Unsupported local embedding provider: {config.provider}")
    model_name = config.model_path or config.model
    model_kwargs = {
        "device": config.device,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
''',
    )
    _write(
        "src/coal_kb/infra/providers/embeddings/factory.py",
        '''"""根据显式模式创建远程或本地 Embedding。"""

from __future__ import annotations

from coal_kb.infra.providers.config import EmbeddingsProviderConfig
from coal_kb.infra.providers.embeddings.local.huggingface import make_local_embeddings
from coal_kb.infra.providers.embeddings.remote.openai_compatible import make_remote_embeddings

EmbeddingsConfig = EmbeddingsProviderConfig


def make_embeddings(config: EmbeddingsProviderConfig):
    """按配置模式创建 Embedding，禁止隐式跨模式回退。"""
    if config.mode == "remote":
        return make_remote_embeddings(config.remote)
    if config.mode == "local":
        return make_local_embeddings(config.local)
    raise ValueError(f"Unsupported embedding mode: {config.mode}")
''',
    )
    _write(
        "src/coal_kb/infra/providers/embeddings/__init__.py",
        '''"""导出 Embedding Provider 正式入口。"""

from coal_kb.infra.providers.config import EmbeddingsProviderConfig as EmbeddingsConfig
from coal_kb.infra.providers.embeddings.factory import make_embeddings

__all__ = ["EmbeddingsConfig", "make_embeddings"]
''',
    )
    _write(
        "src/coal_kb/infra/providers/llm/remote/openai_compatible.py",
        '''"""通过 OpenAI 兼容远程 API 创建 LLM。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from coal_kb.infra.providers.config import LLMProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key

SUPPORTED_REMOTE_LLM_PROVIDERS = {
    "openai",
    "openai_compatible",
    "dashscope",
    "deepseek",
    "siliconflow",
    "moonshot",
    "zhipu",
}


def make_remote_chat_llm(config: LLMProviderConfig) -> ChatOpenAI:
    """创建需要 API Key 的远程聊天模型。"""
    remote = config.remote
    if remote.provider not in SUPPORTED_REMOTE_LLM_PROVIDERS:
        raise ValueError(f"Unsupported remote llm provider: {remote.provider}")
    return ChatOpenAI(
        model=remote.model,
        api_key=resolve_remote_api_key(api_key=remote.api_key, api_key_env=remote.api_key_env),
        base_url=remote.base_url,
        temperature=config.temperature,
        timeout=remote.timeout,
    )
''',
    )
    _write(
        "src/coal_kb/infra/providers/llm/local/openai_compatible.py",
        '''"""连接 vLLM、Ollama 等本地 OpenAI 兼容 LLM 服务。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from coal_kb.infra.providers.config import LLMProviderConfig


def make_local_chat_llm(config: LLMProviderConfig) -> ChatOpenAI:
    """创建不读取远程 API Key 的本地聊天模型客户端。"""
    local = config.local
    if local.provider not in {"openai_compatible", "vllm", "ollama"}:
        raise ValueError(f"Unsupported local llm provider: {local.provider}")
    if not local.base_url:
        raise ValueError("Local LLM base_url is required")
    return ChatOpenAI(
        model=local.model,
        api_key="local-model-no-remote-key",
        base_url=local.base_url,
        temperature=config.temperature,
        timeout=local.timeout,
    )
''',
    )
    _write(
        "src/coal_kb/infra/providers/llm/factory.py",
        '''"""根据显式模式创建远程或本地 LLM。"""

from __future__ import annotations

from coal_kb.infra.providers.config import LLMProviderConfig
from coal_kb.infra.providers.llm.local.openai_compatible import make_local_chat_llm
from coal_kb.infra.providers.llm.remote.openai_compatible import make_remote_chat_llm

LLMConfig = LLMProviderConfig


def make_chat_llm(config: LLMProviderConfig):
    """按配置模式创建 LLM，禁止远程失败后切换本地。"""
    if config.mode == "remote":
        return make_remote_chat_llm(config)
    if config.mode == "local":
        return make_local_chat_llm(config)
    raise ValueError(f"Unsupported llm mode: {config.mode}")
''',
    )
    _write(
        "src/coal_kb/infra/providers/llm/__init__.py",
        '''"""导出 LLM Provider 正式入口。"""

from coal_kb.infra.providers.config import LLMProviderConfig as LLMConfig
from coal_kb.infra.providers.llm.factory import make_chat_llm

__all__ = ["LLMConfig", "make_chat_llm"]
''',
    )
    _write(
        "src/coal_kb/infra/providers/rerank/remote/http.py",
        '''"""通过可配置 HTTP API 执行远程重排序。"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import extract_json_list, resolve_remote_api_key


@dataclass
class RemoteReranker:
    """封装远程重排序连接状态。"""

    config: RemoteProviderConfig

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """调用远程重排序 API 并返回正式排序结果。"""
        if not docs:
            return []
        endpoint = self.config.endpoint or "/rerank"
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": [doc.page_content or "" for doc in docs],
            "top_n": min(top_k, len(docs)),
        }
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {resolve_remote_api_key(api_key=self.config.api_key, api_key_env=self.config.api_key_env)}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = extract_json_list(data, "results", "data")
        indexed_scores: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            score = item.get("relevance_score", item.get("score", 0.0))
            indexed_scores.append((item["index"], float(score)))
        if not indexed_scores:
            scores = extract_json_list(data, "scores")
            indexed_scores = [(index, float(score)) for index, score in enumerate(scores)]
        if not indexed_scores:
            raise ValueError("Remote rerank response does not contain ranked indices or scores")
        ranked = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
        return [docs[index] for index, _ in ranked if 0 <= index < len(docs)][:top_k]
''',
    )
    _write(
        "src/coal_kb/infra/providers/rerank/local/cross_encoder.py",
        '''"""使用本地 CrossEncoder 模型执行重排序。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from coal_kb.infra.providers.config import LocalProviderConfig


@dataclass
class LocalCrossEncoderReranker:
    """延迟加载并复用本地 CrossEncoder。"""

    config: LocalProviderConfig
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.model_path or self.config.model,
                device=None if self.config.device == "auto" else self.config.device,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._model

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """使用本地模型计算候选相关性。"""
        if not docs:
            return []
        scores = self._get_model().predict([(query, doc.page_content or "") for doc in docs])
        ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
''',
    )
    _write(
        "src/coal_kb/infra/providers/rerank/factory.py",
        '''"""根据显式模式创建远程或本地 Reranker。"""

from __future__ import annotations

from coal_kb.infra.providers.config import RerankProviderConfig
from coal_kb.infra.providers.rerank.local.cross_encoder import LocalCrossEncoderReranker
from coal_kb.infra.providers.rerank.remote.http import RemoteReranker


def make_reranker(config: RerankProviderConfig):
    """按配置模式创建 Reranker，失败时不静默改变执行模式。"""
    if config.mode == "remote":
        return RemoteReranker(config.remote)
    if config.mode == "local":
        return LocalCrossEncoderReranker(config.local)
    raise ValueError(f"Unsupported rerank mode: {config.mode}")
''',
    )
    _write(
        "src/coal_kb/infra/providers/rerank/__init__.py",
        '''"""导出 Rerank Provider 正式入口。"""

from coal_kb.infra.providers.config import RerankProviderConfig as RerankConfig
from coal_kb.infra.providers.rerank.factory import make_reranker

__all__ = ["RerankConfig", "make_reranker"]
''',
    )
    _write(
        "src/coal_kb/infra/providers/tokenizers/remote/http.py",
        '''"""通过远程 Tokenizer API 统计 Token。"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key


@dataclass
class RemoteTokenizer:
    """保存远程 Tokenizer API 配置。"""

    config: RemoteProviderConfig

    def count_tokens(self, text: str) -> int:
        """调用远程 tokenize 接口返回精确 Token 数。"""
        if not text:
            return 0
        endpoint = self.config.endpoint or "/tokenize"
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            json={"model": self.config.model, "text": text},
            headers={"Authorization": f"Bearer {resolve_remote_api_key(api_key=self.config.api_key, api_key_env=self.config.api_key_env)}"},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload.get("count"), int):
            return payload["count"]
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("Remote tokenizer response must contain count or tokens")
''',
    )
    _write(
        "src/coal_kb/infra/providers/tokenizers/local/huggingface.py",
        '''"""使用本地 HuggingFace Tokenizer 统计 Token。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coal_kb.infra.providers.config import LocalProviderConfig


@dataclass
class LocalTokenizer:
    """延迟加载并复用本地 Tokenizer。"""

    config: LocalProviderConfig
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model,
                local_files_only=True,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """返回本地模型真实 Token 数。"""
        if not text:
            return 0
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))
''',
    )
    _write(
        "src/coal_kb/infra/providers/tokenizers/factory.py",
        '''"""根据显式模式创建远程或本地 Tokenizer。"""

from __future__ import annotations

from coal_kb.infra.providers.config import TokenizerProviderConfig
from coal_kb.infra.providers.tokenizers.local.huggingface import LocalTokenizer
from coal_kb.infra.providers.tokenizers.remote.http import RemoteTokenizer


def make_tokenizer(config: TokenizerProviderConfig):
    """按配置模式创建 Tokenizer。"""
    if config.mode == "remote":
        return RemoteTokenizer(config.remote)
    if config.mode == "local":
        return LocalTokenizer(config.local)
    raise ValueError(f"Unsupported tokenizer mode: {config.mode}")
''',
    )
    _write(
        "src/coal_kb/infra/providers/tokenizers/__init__.py",
        '''"""导出 Tokenizer Provider 正式入口。"""

from coal_kb.infra.providers.config import TokenizerProviderConfig as TokenizerConfig
from coal_kb.infra.providers.tokenizers.factory import make_tokenizer

__all__ = ["TokenizerConfig", "make_tokenizer"]
''',
    )
    _write(
        "src/coal_kb/infra/providers/__init__.py",
        '''"""集中导出模型 Provider 配置与工厂。"""

from coal_kb.infra.providers.config import (
    EmbeddingsProviderConfig,
    LLMProviderConfig,
    LocalProviderConfig,
    RemoteProviderConfig,
    RerankProviderConfig,
    TokenizerProviderConfig,
)

__all__ = [
    "EmbeddingsProviderConfig",
    "LLMProviderConfig",
    "LocalProviderConfig",
    "RemoteProviderConfig",
    "RerankProviderConfig",
    "TokenizerProviderConfig",
]
''',
    )

    models_path = ROOT / "src/coal_kb/infra/config/models.py"
    models_text = models_path.read_text(encoding="utf-8")
    models_text = models_text.replace("from typing import Optional\n", "from typing import Optional\n\nfrom coal_kb.infra.providers.config import EmbeddingsProviderConfig, LLMProviderConfig, RerankProviderConfig, TokenizerProviderConfig\n")
    models_text = re.sub(r"\nclass RerankConfig\(BaseModel\):.*?(?=\nclass LoggingConfig)", "\n", models_text, flags=re.S)
    models_text = re.sub(r"\nclass LLMConfig\(BaseModel\):.*?(?=\nclass RemoteEmbeddingsConfig)", "\n", models_text, flags=re.S)
    models_text = re.sub(r"\nclass RemoteEmbeddingsConfig\(BaseModel\):.*?(?=\nclass AppConfig)", "\n", models_text, flags=re.S)
    models_text = models_text.replace("    rerank: RerankConfig = Field(default_factory=RerankConfig)\n", "    tokenizer: TokenizerProviderConfig\n    embeddings: EmbeddingsProviderConfig\n    rerank: RerankProviderConfig\n    llm: LLMProviderConfig\n")
    models_text = models_text.replace("\n    llm: LLMConfig = Field(default_factory=LLMConfig)\n    embeddings: RemoteEmbeddingsConfig = Field(default_factory=RemoteEmbeddingsConfig)\n", "\n")
    models_path.write_text(models_text, encoding="utf-8")

    _write(
        "configs/app.yaml",
        '''paths:
  raw_pdfs_dir: "data/raw_pdfs"
  raw_docs_dir: "data/raw_docs"
  interim_dir: "data/interim"
  artifacts_dir: "data/artifacts"
  chroma_dir: "storage/chroma_db"
  sqlite_path: "storage/expert.db"
  manifest_path: "storage/manifest.json"

model_versions:
  embedding_version: "v1"

chunking:
  strategy: markdown_hierarchical_semantic
  max_parent_tokens: 600
  max_child_tokens: 200
  overlap_tokens: 20
  similarity_threshold: 0.72
  heading_max_depth: 4
  embedding_backend: configured
  chunk_size: 900
  chunk_overlap: 120
  profile_by_section:
    results: {chunk_size: 900, chunk_overlap: 150}
    discussion: {chunk_size: 900, chunk_overlap: 150}
    methods: {chunk_size: 650, chunk_overlap: 120}
    conditions: {chunk_size: 650, chunk_overlap: 120}
    unknown: {chunk_size: 750, chunk_overlap: 120}

pdf_markdown:
  enabled: true
  heading_max_depth: 4
  two_column_mode: auto
  drop_headers_footers: true
  min_heading_font_ratio: 1.15

chroma:
  collection_name: "coal_gasification_papers"

logging:
  level: "INFO"

elastic:
  host: "http://localhost:9200"
  index_prefix: "coal_kb_chunks"
  alias_current: "coal_kb_chunks_current"
  alias_prev: "coal_kb_chunks_prev"
  verify_certs: false
  timeout_s: 180
  bulk_chunk_size: 100
  enable_icu_analyzer: true

ingestion:
  include_exts: ["pdf", "txt", "md", "docx"]
  drop_reference_like_unknown: true
  drop_sections: [references, acknowledgements, contents, appendix]

query_rewrite:
  enable_llm: false

tenancy:
  enabled: false
  default_tenant_id: "default"
  enforce_tenant_filter: true

registry:
  sqlite_path: "storage/kb.db"

backend: "elastic"

retrieval:
  k: 10
  rrf_k: 60
  max_per_source: 4
  max_relax_steps: 3
  range_expand_schedule: [0.05, 0.1, 0.2]
  mode: "broad"
  rerank_enabled: true
  rerank_top_n: 10
  two_stage:
    enabled: true
    parent_k_candidates: 200
    parent_k_final: 80
    max_parents: 80
    child_k_candidates: 200
    child_k_final: 60
    allow_relax_in_stage2: true

tokenizer:
  mode: local
  remote:
    provider: openai_compatible
    base_url: "https://api.example.com/v1"
    api_key_env: "TOKENIZER_API_KEY"
    model: "remote-tokenizer"
    endpoint: "/tokenize"
  local:
    provider: huggingface
    model: "Qwen/Qwen3-8B"
    model_path: null
    device: auto
    trust_remote_code: true

embeddings:
  mode: remote
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "text-embedding-v4"
    dimensions: 1024
  local:
    provider: huggingface
    model: "BAAI/bge-m3"
    model_path: null
    device: auto
    dimensions: 1024

rerank:
  mode: remote
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "qwen3-rerank"
    endpoint: "/text-rerank/text-rerank"
    timeout: 60
  local:
    provider: cross_encoder
    model: "BAAI/bge-reranker-base"
    model_path: null
    device: auto

llm:
  mode: remote
  temperature: 0.0
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "qwen3.5-flash"
    timeout: 60
  local:
    provider: vllm
    base_url: "http://127.0.0.1:8001/v1"
    model: "Qwen/Qwen3-8B"
    model_path: null
    device: auto
    timeout: 60
''',
    )

    ask_path = ROOT / "src/coal_kb/application/ask.py"
    ask_text = ask_path.read_text(encoding="utf-8")
    ask_text = ask_text.replace("    from coal_kb.infra.providers.llm import LLMConfig\n", "    from coal_kb.infra.providers.llm import LLMConfig\n")
    ask_text = ask_text.replace("            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),", "            embeddings_cfg=cfg.embeddings,")
    ask_text = ask_text.replace("            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),", "            embeddings_cfg=cfg.embeddings,")
    ask_text = ask_text.replace("        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()) if active_backend == \"elastic\" else None,", "        embeddings_cfg=cfg.embeddings if active_backend == \"elastic\" else None,")
    ask_text = ask_text.replace("    reranker = make_reranker(cfg) if active_rerank else None", "    reranker = make_reranker(cfg.rerank) if active_rerank else None")
    ask_text = ask_text.replace("        final_provider = cfg.llm.provider", "        final_provider = cfg.llm.provider")
    ask_text = ask_text.replace("        llm_config = LLMConfig(**{**cfg.llm.model_dump(), \"provider\": final_provider})", "        llm_config = cfg.llm.model_copy(deep=True)\n        llm_config.active.provider = final_provider")
    ask_text = ask_text.replace("        context_builder=ContextBuilder(),", "        context_builder=ContextBuilder(token_counter=make_tokenizer(cfg.tokenizer).count_tokens),")
    ask_text = ask_text.replace("    from coal_kb.infra.providers.rerank import make_reranker\n", "    from coal_kb.infra.providers.rerank import make_reranker\n    from coal_kb.infra.providers.tokenizers import make_tokenizer\n")
    ask_path.write_text(ask_text, encoding="utf-8")

    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        text = text.replace("EmbeddingsConfig(**cfg.embeddings.model_dump())", "cfg.embeddings")
        path.write_text(text, encoding="utf-8")

    _write(
        "src/coal_kb/context/budgeting.py",
        '''"""负责上下文真实 Token 计数和证据预算选择。"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

TokenCounter = Callable[[str], int]


def select_with_budget(
    docs: list[Document],
    *,
    max_chunks: int,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> tuple[list[Document], int, int]:
    """使用当前模型 Tokenizer 选择预算内证据。"""
    selected: list[Document] = []
    dropped_budget = 0
    token_total = 0
    for doc in docs:
        if len(selected) >= max_chunks:
            dropped_budget += 1
            continue
        document_tokens = count_tokens(doc.page_content or "")
        if selected and max_tokens and token_total + document_tokens > max_tokens:
            dropped_budget += 1
            continue
        selected.append(doc)
        token_total += document_tokens
    return selected, dropped_budget, token_total
''',
    )
    context_path = ROOT / "src/coal_kb/context/service.py"
    context_text = context_path.read_text(encoding="utf-8")
    context_text = context_text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom collections.abc import Callable\n")
    context_text = context_text.replace("class ContextBuilder:\n    \"\"\"保持原 ContextBuilder 接口和证据编号顺序。\"\"\"\n\n    def build", "class ContextBuilder:\n    \"\"\"使用注入的模型 Tokenizer 构建预算内上下文。\"\"\"\n\n    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:\n        from coal_kb.tokenization import count_tokens\n\n        self._token_counter = token_counter or count_tokens\n\n    def build")
    context_text = context_text.replace("            max_tokens=max_tokens,\n        )", "            max_tokens=max_tokens,\n            count_tokens=self._token_counter,\n        )")
    context_path.write_text(context_text, encoding="utf-8")

    _write(
        "tests/test_provider_runtime.py",
        '''"""验证远程与本地 Provider 隔离及 Token 预算接入。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.context.budgeting import select_with_budget
from coal_kb.infra.config import load_config
from coal_kb.infra.providers.rerank.factory import make_reranker
from coal_kb.infra.providers.tokenizers.factory import make_tokenizer


def test_provider_modes_are_explicit_and_separate() -> None:
    load_config.cache_clear()
    cfg = load_config()
    assert cfg.embeddings.mode in {"remote", "local"}
    assert cfg.llm.mode in {"remote", "local"}
    assert cfg.rerank.mode in {"remote", "local"}
    assert cfg.tokenizer.mode in {"remote", "local"}
    assert not hasattr(cfg.llm.local, "api_key")
    assert not hasattr(cfg.rerank.local, "api_key_env")


def test_reranker_factory_uses_configured_mode() -> None:
    load_config.cache_clear()
    cfg = load_config()
    cfg.rerank.mode = "local"
    reranker = make_reranker(cfg.rerank)
    assert reranker.__class__.__name__ == "LocalCrossEncoderReranker"


def test_token_budget_uses_injected_counter() -> None:
    docs = [Document(page_content="one"), Document(page_content="two")]
    selected, dropped, total = select_with_budget(
        docs,
        max_chunks=2,
        max_tokens=3,
        count_tokens=lambda text: 2,
    )
    assert len(selected) == 1
    assert dropped == 1
    assert total == 2


def test_local_tokenizer_does_not_load_until_used() -> None:
    load_config.cache_clear()
    cfg = load_config()
    cfg.tokenizer.mode = "local"
    tokenizer = make_tokenizer(cfg.tokenizer)
    assert tokenizer.__class__.__name__ == "LocalTokenizer"
    assert tokenizer._tokenizer is None
''',
    )

    quality = ROOT / "scripts/quality/config.sh"
    quality_text = quality.read_text(encoding="utf-8")
    quality_text = quality_text.replace(
        '  "$REPO_ROOT/tests/test_config_consistency.py"\n',
        '  "$REPO_ROOT/tests/test_config_consistency.py"\n  "$REPO_ROOT/tests/test_provider_runtime.py"\n',
    )
    quality.write_text(quality_text, encoding="utf-8")

    api_models = ROOT / "src/coal_kb/interfaces/api/models.py"
    api_text = api_models.read_text(encoding="utf-8")
    api_text = api_text.replace(
        '    llm_provider: str = "none"\n    api_key: Optional[str] = Field(default=None, min_length=1)\n    provider_base_url: Optional[str] = Field(default=None, min_length=1)\n    llm_model: Optional[str] = Field(default=None, min_length=1)\n    embedding_model: Optional[str] = Field(default=None, min_length=1)\n',
        '    tokenizer_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")\n    tokenizer_provider: Optional[str] = None\n    tokenizer_base_url: Optional[str] = None\n    tokenizer_api_key: Optional[str] = None\n    tokenizer_model: Optional[str] = None\n    embedding_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")\n    embedding_provider: Optional[str] = None\n    embedding_base_url: Optional[str] = None\n    embedding_api_key: Optional[str] = None\n    embedding_model: Optional[str] = None\n    rerank_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")\n    rerank_provider: Optional[str] = None\n    rerank_base_url: Optional[str] = None\n    rerank_api_key: Optional[str] = None\n    rerank_model: Optional[str] = None\n    llm_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")\n    llm_provider: str = "none"\n    llm_base_url: Optional[str] = None\n    llm_api_key: Optional[str] = None\n    llm_model: Optional[str] = None\n',
    )
    api_text = re.sub(
        r'class SettingsDefaultsResponse\(BaseModel\):.*',
        '''class SettingsDefaultsResponse(BaseModel):
    api_base_url: str = ""
    tokenizer: Dict[str, Any]
    embeddings: Dict[str, Any]
    rerank_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    backend: str
    mode: str
    k: int
    rerank: bool
    llm: bool = False
    debug: bool = False
    backend_options: List[str] = Field(default_factory=list)
    mode_options: List[str] = Field(default_factory=list)
    remote_provider_options: List[str] = Field(default_factory=list)
    local_provider_options: Dict[str, List[str]] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
''',
        api_text,
        flags=re.S,
    )
    api_models.write_text(api_text, encoding="utf-8")

    _write(
        "src/coal_kb/interfaces/api/runtime_overrides.py",
        '''"""将请求级 Provider 设置覆盖到配置副本。"""

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
''',
    )

    html_path = ROOT / "src/coal_kb/interfaces/web/static/index.html"
    html = html_path.read_text(encoding="utf-8")
    old_fields = '''            <label class="field">
              <span>Provider 基础地址</span>
              <input id="setting-provider-base-url" type="text" placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1" />
            </label>
            <label class="field">
              <span>API Key</span>
              <input id="setting-api-key" type="password" placeholder="可选，每次请求时覆盖" />
            </label>
            <label class="field">
              <span>LLM 提供商</span>
              <select id="setting-llm-provider"></select>
            </label>
            <label class="field">
              <span>LLM 模型</span>
              <input id="setting-llm-model" type="text" placeholder="qwen-plus" />
            </label>
            <label class="field">
              <span>嵌入模型</span>
              <input id="setting-embedding-model" type="text" placeholder="text-embedding-v3" />
            </label>'''
    new_fields = '''            <label class="field"><span>Tokenizer 模式</span><select id="setting-tokenizer-mode"><option value="local">本地</option><option value="remote">远程</option></select></label>
            <label class="field"><span>Tokenizer Provider</span><input id="setting-tokenizer-provider" type="text" /></label>
            <label class="field"><span>Tokenizer 地址</span><input id="setting-tokenizer-base-url" type="text" /></label>
            <label class="field"><span>Tokenizer API Key</span><input id="setting-tokenizer-api-key" type="password" /></label>
            <label class="field"><span>Tokenizer 模型</span><input id="setting-tokenizer-model" type="text" /></label>
            <label class="field"><span>Embedding 模式</span><select id="setting-embedding-mode"><option value="remote">远程</option><option value="local">本地</option></select></label>
            <label class="field"><span>Embedding Provider</span><input id="setting-embedding-provider" type="text" /></label>
            <label class="field"><span>Embedding 地址</span><input id="setting-embedding-base-url" type="text" /></label>
            <label class="field"><span>Embedding API Key</span><input id="setting-embedding-api-key" type="password" /></label>
            <label class="field"><span>Embedding 模型</span><input id="setting-embedding-model" type="text" /></label>
            <label class="field"><span>Rerank 模式</span><select id="setting-rerank-mode"><option value="remote">远程</option><option value="local">本地</option></select></label>
            <label class="field"><span>Rerank Provider</span><input id="setting-rerank-provider" type="text" /></label>
            <label class="field"><span>Rerank 地址</span><input id="setting-rerank-base-url" type="text" /></label>
            <label class="field"><span>Rerank API Key</span><input id="setting-rerank-api-key" type="password" /></label>
            <label class="field"><span>Rerank 模型</span><input id="setting-rerank-model" type="text" /></label>
            <label class="field"><span>LLM 模式</span><select id="setting-llm-mode"><option value="remote">远程</option><option value="local">本地</option></select></label>
            <label class="field"><span>LLM Provider</span><input id="setting-llm-provider" type="text" /></label>
            <label class="field"><span>LLM 地址</span><input id="setting-llm-base-url" type="text" /></label>
            <label class="field"><span>LLM API Key</span><input id="setting-llm-api-key" type="password" /></label>
            <label class="field"><span>LLM 模型</span><input id="setting-llm-model" type="text" /></label>'''
    if old_fields not in html:
        raise ValueError("UI settings fields changed")
    html_path.write_text(html.replace(old_fields, new_fields), encoding="utf-8")

    js_path = ROOT / "src/coal_kb/interfaces/web/static/app.js"
    js = js_path.read_text(encoding="utf-8")
    js = js.replace('  settingProviderBaseUrl: $("setting-provider-base-url"),\n  settingApiKey: $("setting-api-key"),\n  settingLlmProvider: $("setting-llm-provider"),\n  settingLlmModel: $("setting-llm-model"),\n  settingEmbeddingModel: $("setting-embedding-model"),\n', '''  settingTokenizerMode: $("setting-tokenizer-mode"),
  settingTokenizerProvider: $("setting-tokenizer-provider"),
  settingTokenizerBaseUrl: $("setting-tokenizer-base-url"),
  settingTokenizerApiKey: $("setting-tokenizer-api-key"),
  settingTokenizerModel: $("setting-tokenizer-model"),
  settingEmbeddingMode: $("setting-embedding-mode"),
  settingEmbeddingProvider: $("setting-embedding-provider"),
  settingEmbeddingBaseUrl: $("setting-embedding-base-url"),
  settingEmbeddingApiKey: $("setting-embedding-api-key"),
  settingEmbeddingModel: $("setting-embedding-model"),
  settingRerankMode: $("setting-rerank-mode"),
  settingRerankProvider: $("setting-rerank-provider"),
  settingRerankBaseUrl: $("setting-rerank-base-url"),
  settingRerankApiKey: $("setting-rerank-api-key"),
  settingRerankModel: $("setting-rerank-model"),
  settingLlmMode: $("setting-llm-mode"),
  settingLlmProvider: $("setting-llm-provider"),
  settingLlmBaseUrl: $("setting-llm-base-url"),
  settingLlmApiKey: $("setting-llm-api-key"),
  settingLlmModel: $("setting-llm-model"),
''')
    js = js.replace('    llm_provider: s.llmProvider || "none", api_key: s.apiKey || null,\n    provider_base_url: s.providerBaseUrl || null, llm_model: s.llmModel || null, embedding_model: s.embeddingModel || null,', '''    tokenizer_mode: s.tokenizerMode || null,
    tokenizer_provider: s.tokenizerProvider || null,
    tokenizer_base_url: s.tokenizerBaseUrl || null,
    tokenizer_api_key: s.tokenizerApiKey || null,
    tokenizer_model: s.tokenizerModel || null,
    embedding_mode: s.embeddingMode || null,
    embedding_provider: s.embeddingProvider || null,
    embedding_base_url: s.embeddingBaseUrl || null,
    embedding_api_key: s.embeddingApiKey || null,
    embedding_model: s.embeddingModel || null,
    rerank_mode: s.rerankMode || null,
    rerank_provider: s.rerankProvider || null,
    rerank_base_url: s.rerankBaseUrl || null,
    rerank_api_key: s.rerankApiKey || null,
    rerank_model: s.rerankModel || null,
    llm_mode: s.llmMode || null,
    llm_provider: s.llmProvider || "none",
    llm_base_url: s.llmBaseUrl || null,
    llm_api_key: s.llmApiKey || null,
    llm_model: s.llmModel || null,''')
    js = js.replace('providerBaseUrl', 'llmBaseUrl').replace('apiKey', 'llmApiKey')
    js_path.write_text(js, encoding="utf-8")


if __name__ == "__main__":
    process()
