"""使用版本化知识图谱抽取和传播执行可解释证据重排。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from coal_kb.research.graph_extraction import KnowledgeGraphExtractor
from coal_kb.research.graph_schema import GraphNodeType, GraphRelationType, KnowledgeGraph
from coal_kb.research.models import RouteResult
from coal_kb.utils.documents import deduplicate_documents, document_key

BaseRoute = Callable[[], RouteResult]


@dataclass(frozen=True)
class GraphEdge:
    """保存投影到证据节点之间的关系权重和理由。"""

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
    """在基础路线证据内抽取知识图谱并执行受控关系传播。"""

    seed_count: int = 3
    max_edges: int = 80
    use_shared_terms: bool = True
    use_entities: bool = True
    use_claims: bool = True
    entity_weight: float = 0.45
    claim_weight: float = 0.2

    def process(self, base_route: BaseRoute) -> RouteResult:
        base = base_route()
        documents = deduplicate_documents(base.documents)
        extractor = KnowledgeGraphExtractor(
            use_shared_terms=self.use_shared_terms,
            use_entities=self.use_entities,
            use_claims=self.use_claims,
        )
        graph = extractor.process(documents)
        edges = self._project_edges(graph)[: self.max_edges]
        ranked, scores = self._rank(documents, graph)
        graph_payload = graph.to_dict()
        trace = {
            "route": "graph",
            "base": base.trace,
            "graph": {
                "schema_version": graph.schema_version,
                "node_count": len(documents),
                "typed_node_count": len(graph.nodes),
                "edge_count": len(edges),
                "typed_relation_count": len(graph.relations),
                "seed_chunk_ids": [document_key(item) for item in documents[: self.seed_count]],
                "selected_chunk_ids": [document_key(item) for item in ranked],
                "scores": scores,
                "edges": [edge.to_dict() for edge in edges],
                "statistics": graph_payload["statistics"],
                "nodes": graph_payload["nodes"],
                "relations": graph_payload["relations"],
                "configuration": {
                    "use_shared_terms": self.use_shared_terms,
                    "use_entities": self.use_entities,
                    "use_claims": self.use_claims,
                    "entity_weight": self.entity_weight,
                    "claim_weight": self.claim_weight,
                },
            },
        }
        return RouteResult(documents=ranked, trace=trace)

    @staticmethod
    def _project_edges(graph: KnowledgeGraph) -> list[GraphEdge]:
        grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for relation in graph.relations.values():
            source = graph.nodes[relation.source]
            target = graph.nodes[relation.target]
            if source.node_type is not GraphNodeType.EVIDENCE or target.node_type is not GraphNodeType.EVIDENCE:
                continue
            source_chunk = str(source.properties.get("chunk_id") or source.node_id)
            target_chunk = str(target.properties.get("chunk_id") or target.node_id)
            if source_chunk >= target_chunk:
                continue
            grouped.setdefault((source_chunk, target_chunk), []).append(
                (relation.relation_type.value, relation.weight * relation.confidence)
            )
        edges = [
            GraphEdge(
                source=source,
                target=target,
                weight=round(sum(weight for _, weight in reasons), 4),
                reasons=tuple(sorted(reason for reason, _ in reasons)),
            )
            for (source, target), reasons in grouped.items()
        ]
        return sorted(edges, key=lambda edge: (-edge.weight, edge.source, edge.target))

    def _rank(self, documents: list[Document], graph: KnowledgeGraph) -> tuple[list[Document], dict[str, float]]:
        evidence_by_chunk = {
            str(node.properties.get("chunk_id")): node.node_id
            for node in graph.nodes_of_type(GraphNodeType.EVIDENCE)
        }
        original_rank = {document_key(document): index for index, document in enumerate(documents)}
        seed_chunks = set(list(original_rank)[: self.seed_count])
        seed_nodes = {evidence_by_chunk[chunk] for chunk in seed_chunks if chunk in evidence_by_chunk}
        scores = {key: 1.0 / (rank + 1) for key, rank in original_rank.items()}

        for relation in graph.relations.values():
            source = graph.nodes[relation.source]
            target = graph.nodes[relation.target]
            value = relation.weight * relation.confidence / 4.0
            if source.node_type is GraphNodeType.EVIDENCE and target.node_type is GraphNodeType.EVIDENCE:
                target_chunk = str(target.properties.get("chunk_id"))
                if relation.source in seed_nodes and target_chunk in scores:
                    scores[target_chunk] += value

        mentions: dict[str, list[tuple[str, float]]] = {}
        claim_counts: dict[str, int] = {}
        for relation in graph.relations.values():
            if relation.relation_type is GraphRelationType.MENTIONS:
                mentions.setdefault(relation.target, []).append((relation.source, relation.confidence))
            elif relation.relation_type is GraphRelationType.SUPPORTS:
                claim_counts[relation.source] = claim_counts.get(relation.source, 0) + 1
        if self.use_entities:
            for evidence_mentions in mentions.values():
                if not any(evidence_id in seed_nodes for evidence_id, _ in evidence_mentions):
                    continue
                for evidence_id, confidence in evidence_mentions:
                    node = graph.nodes[evidence_id]
                    chunk_id = str(node.properties.get("chunk_id"))
                    if chunk_id in scores and evidence_id not in seed_nodes:
                        scores[chunk_id] += self.entity_weight * confidence
        if self.use_claims:
            for evidence_id, count in claim_counts.items():
                node = graph.nodes[evidence_id]
                chunk_id = str(node.properties.get("chunk_id"))
                if chunk_id in scores:
                    scores[chunk_id] += min(3, count) * self.claim_weight / 10.0

        ranked = sorted(
            documents,
            key=lambda document: (
                -scores[document_key(document)],
                original_rank[document_key(document)],
            ),
        )
        return ranked, {key: round(value, 6) for key, value in scores.items()}
