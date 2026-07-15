"""导出 Rerank Provider 正式入口。"""

from coal_kb.infra.providers.config import RerankProviderConfig as RerankConfig
from coal_kb.infra.providers.rerank.factory import make_reranker

__all__ = ["RerankConfig", "make_reranker"]
