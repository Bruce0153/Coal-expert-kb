"""Backward-compatible chat-model provider imports."""

from coal_kb.infra.providers.llm import LLMConfig, make_chat_llm

__all__ = ["LLMConfig", "make_chat_llm"]
