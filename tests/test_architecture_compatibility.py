from __future__ import annotations

from coal_kb import settings
from coal_kb.core.models import query as core_query
from coal_kb.embeddings import factory as legacy_embeddings
from coal_kb.infra import config
from coal_kb.infra.providers import embeddings, llm, rerank
from coal_kb.llm import factory as legacy_llm
from coal_kb.query import plan as legacy_query
from coal_kb.retrieval import rerank as legacy_rerank


def test_legacy_configuration_exports_are_canonical_objects() -> None:
    assert settings.AppConfig is config.AppConfig
    assert settings.load_config is config.load_config
    assert settings._load_yaml_unique_keys is config._load_yaml_unique_keys


def test_legacy_query_plan_exports_are_canonical_objects() -> None:
    assert legacy_query.QueryPlan is core_query.QueryPlan
    assert legacy_query.Constraint is core_query.Constraint


def test_legacy_provider_exports_are_canonical_objects() -> None:
    assert legacy_embeddings.EmbeddingsConfig is embeddings.EmbeddingsConfig
    assert legacy_embeddings.make_embeddings is embeddings.make_embeddings
    assert legacy_llm.LLMConfig is llm.LLMConfig
    assert legacy_llm.make_chat_llm is llm.make_chat_llm
    assert legacy_rerank.DashScopeReranker is rerank.DashScopeReranker
    assert legacy_rerank.make_reranker is rerank.make_reranker
