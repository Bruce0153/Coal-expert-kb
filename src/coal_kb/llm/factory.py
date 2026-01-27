from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.0
    timeout: int = 60


def make_chat_llm(cfg: LLMConfig) -> ChatOpenAI:
    if cfg.provider not in ("dashscope", "openai_compatible", "openai"):
        raise ValueError(f"Unsupported llm.provider: {cfg.provider}")

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {cfg.api_key_env}")

    # DashScope is OpenAI-compatible mode (chat/completions under base_url)
    return ChatOpenAI(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )
