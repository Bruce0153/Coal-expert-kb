"""验证远程与本地 Provider 隔离及 Token 预算接入。"""

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
