"""定义可版本化、可验证和可序列化的研究知识图谱协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

GRAPH_SCHEMA_VERSION = "coal-knowledge-graph.v1"


class GraphNodeType(str, Enum):
    """知识图谱允许的节点类型。"""

    EVIDENCE = "evidence"
    ENTITY = "entity"
    CLAIM = "claim"


class GraphRelationType(str, Enum):
    """知识图谱允许的有向关系类型。"""

    SAME_PARENT = "same_parent"
    SAME_SOURCE = "same_source"
    SAME_SECTION = "same_section"
    SHARED_TERMS = "shared_terms"
    MENTIONS = "mentions"
    CO_OCCURS = "co_occurs"
    SUPPORTS = "supports"
    ABOUT = "about"


@dataclass(frozen=True)
class GraphNode:
    """保存稳定节点标识、类型、标签与来源属性。"""

    node_id: str
    node_type: GraphNodeType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphRelation:
    """保存有向关系、权重、置信度和证据来源。"""

    source: str
    target: str
    relation_type: GraphRelationType
    weight: float
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation_type.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "weight": round(self.weight, 6),
            "confidence": round(self.confidence, 6),
            "provenance": self.provenance,
        }


@dataclass
class KnowledgeGraph:
    """维护节点和关系完整性，并提供稳定序列化结果。"""

    schema_version: str = GRAPH_SCHEMA_VERSION
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], GraphRelation] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"Conflicting graph node: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_relation(self, relation: GraphRelation) -> None:
        if relation.source not in self.nodes or relation.target not in self.nodes:
            raise ValueError("Graph relation endpoints must exist before the relation is added")
        if not 0.0 <= relation.confidence <= 1.0:
            raise ValueError("Graph relation confidence must be between 0 and 1")
        if relation.weight < 0:
            raise ValueError("Graph relation weight cannot be negative")
        key = relation.key()
        existing = self.relations.get(key)
        if existing is None or relation.weight * relation.confidence > existing.weight * existing.confidence:
            self.relations[key] = relation

    def nodes_of_type(self, node_type: GraphNodeType) -> list[GraphNode]:
        return sorted(
            (node for node in self.nodes.values() if node.node_type is node_type),
            key=lambda node: node.node_id,
        )

    def relations_of_type(self, relation_type: GraphRelationType) -> list[GraphRelation]:
        return sorted(
            (relation for relation in self.relations.values() if relation.relation_type is relation_type),
            key=lambda relation: relation.key(),
        )

    def validate(self) -> None:
        if self.schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported graph schema version: {self.schema_version}")
        for relation in self.relations.values():
            if relation.source not in self.nodes or relation.target not in self.nodes:
                raise ValueError(f"Dangling graph relation: {relation.key()}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in sorted(self.nodes.values(), key=lambda item: item.node_id)],
            "relations": [
                relation.to_dict()
                for relation in sorted(self.relations.values(), key=lambda item: item.key())
            ],
            "statistics": {
                "node_count": len(self.nodes),
                "relation_count": len(self.relations),
                "node_types": {
                    node_type.value: len(self.nodes_of_type(node_type))
                    for node_type in GraphNodeType
                },
                "relation_types": {
                    relation_type.value: len(self.relations_of_type(relation_type))
                    for relation_type in GraphRelationType
                },
            },
        }
