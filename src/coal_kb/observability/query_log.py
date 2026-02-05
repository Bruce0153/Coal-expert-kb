from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QueryLog:
    query: str
    plan: Dict[str, Any]
    filters: Dict[str, Any]
    k: int
    latency_ms: float
    relax_steps: List[Any] = field(default_factory=list)
    top_sources: List[str] = field(default_factory=list)
    top_chunk_ids: List[str] = field(default_factory=list)
    answer_stats: Dict[str, Any] = field(default_factory=dict)
