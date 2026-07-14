"""构建稳定的检索 trace 摘要。"""

from __future__ import annotations

from typing import Any


def build_retrieval_trace_summary(
    *,
    retrieval_query: str,
    history_used: bool,
    history_reason: str,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """提取 API 和日志共同使用的 trace 字段。"""
    return {
        "retrieval_query": retrieval_query,
        "history_used": history_used,
        "history_reason": history_reason,
        "vector_candidates": trace.get("vector_candidates"),
        "postfiltered_count": trace.get("postfiltered_count"),
        "source_distribution": trace.get("source_distribution"),
        "heading_distribution": trace.get("heading_distribution"),
    }
