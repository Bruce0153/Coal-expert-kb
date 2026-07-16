"""在已召回证据之间构建可解释关系并执行稳定重排。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from coal_kb.research.models import RouteResult
from coal_kb.utils.documents import deduplicate_documents, document_key

BaseRoute = Callable[[], RouteResult]


@dataclass(frozen=True)
class GraphEdge:
    """保存两条证据之间的关系权重和理由。"""

    source: str
    target: str
    weight: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "reasons": list(self.reasons),
        }


@dataclass
class GraphRoute:
    """只重排基础路线返回的证据，不扩展到未召回内容。"""

    seed_count: int = 3
    max_edges: int = 80

    def process(self, base_route: BaseRoute) -> RouteResult:
        base = base_route()
        documents = deduplicate_documents(base.documents)
        edges = self._build_edges(documents)
        ranked, scores = self._rank(documents, edges)
        trace = {
            "route": "graph",
            "base": base.trace,
            "graph": {
                "node_count": len(documents),
                "edge_count": len(edges),
                "seed_chunk_ids": [document_key(item) for item in documents[: self.seed_count]],
                "selected_chunk_ids": [document_key(item) for item in ranked],
                "scores": scores,
                "edges": [edge.to_dict() for edge in edges],
            },
        }
        return RouteResult(documents=ranked, trace=trace)

    def _build_edges(self, documents: list[Document]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        terms = [self._terms(document.page_content) for document in documents]
        for left_index, left in enumerate(documents):
            for right_index in range(left_index + 1, len(documents)):
                right = documents[right_index]
                reasons: list[str] = []
                weight = 0.0
                left_meta = left.metadata or {}
                right_meta = right.metadata or {}
                if left_meta.get("parent_id") and left_meta.get("parent_id") == right_meta.get("parent_id"):
                    weight += 3.0
                    reasons.append("same_parent")
                if left_meta.get("source_file") and left_meta.get("source_file") == right_meta.get("source_file"):
                    weight += 1.0
                    reasons.append("same_source")
                left_heading = left_meta.get("heading_path") or left_meta.get("section")
                right_heading = right_meta.get("heading_path") or right_meta.get("section")
                if left_heading and left_heading == right_heading:
                    weight += 1.0
                    reasons.append("same_section")
                overlap = terms[left_index] & terms[right_index]
                if overlap:
                    bridge_weight = min(2.0, len(overlap) / 3.0)
                    weight += bridge_weight
                    reasons.append("shared_terms")
                if weight > 0:
                    edges.append(
                        GraphEdge(
                            source=document_key(left),
                            target=document_key(right),
                            weight=round(weight, 4),
                            reasons=tuple(reasons),
                        )
                    )
        return sorted(edges, key=lambda edge: (-edge.weight, edge.source, edge.target))[: self.max_edges]

    def _rank(
        self,
        documents: list[Document],
        edges: list[GraphEdge],
    ) -> tuple[list[Document], dict[str, float]]:
        original_rank = {document_key(document): index for index, document in enumerate(documents)}
        seeds = set(list(original_rank)[: self.seed_count])
        scores = {
            key: 1.0 / (rank + 1)
            for key, rank in original_rank.items()
        }
        for edge in edges:
            if edge.source in seeds:
                scores[edge.target] += edge.weight / 4.0
            if edge.target in seeds:
                scores[edge.source] += edge.weight / 4.0
        ranked = sorted(
            documents,
            key=lambda document: (
                -scores[document_key(document)],
                original_rank[document_key(document)],
            ),
        )
        return ranked, {key: round(value, 6) for key, value in scores.items()}

    @staticmethod
    def _terms(text: str) -> set[str]:
        english = re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
        blocked = {"研究", "结果", "实验", "条件", "过程", "方法", "影响"}
        return {term for term in english + chinese if term not in blocked}
