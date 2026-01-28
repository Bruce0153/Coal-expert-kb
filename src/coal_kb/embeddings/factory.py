from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    dimensions: Optional[int] = None


def make_embeddings(cfg: EmbeddingsConfig) -> OpenAIEmbeddings:
    if cfg.provider not in ("dashscope", "openai_compatible", "openai"):
        raise ValueError(f"Unsupported embeddings.provider: {cfg.provider}")

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {cfg.api_key_env}")

    kwargs = {}

    # DashScope embedding supports configurable dimensions for text-embedding-v4/v3
    if cfg.dimensions is not None:
        kwargs["dimensions"] = int(cfg.dimensions)

    if cfg.provider == "dashscope":
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 10

    return OpenAIEmbeddings(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        **kwargs,
    )

