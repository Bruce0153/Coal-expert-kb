"""从标准 Document 证据中确定性抽取实体、Claim 和关系。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from langchain_core.documents import Document

from coal_kb.research.graph_schema import (
    GraphNode,
    GraphNodeType,
    GraphRelation,
    GraphRelationType,
    KnowledgeGraph,
)
from coal_kb.utils.documents import document_key

_ENGLISH_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}[0-9A-Z-]*)\b")
_CHINESE_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
_SENTENCE_RE = re.compile(r"(?<=[。！？!?;；])\s+|\n+")
_CLAIM_MARKERS = re.compile(
    r"导致|影响|促进|抑制|提高|降低|表明|说明|由于|因此|相关|cause|lead|increase|decrease|affect|promote|inhibit|indicate",
    re.IGNORECASE,
)
_BLOCKED_TERMS = {
    "研究",
    "结果",
    "实验",
    "条件",
    "过程",
    "方法",
    "影响",
    "分析",
    "数据",
    "模型",
    "系统",
    "本文",
    "可以",
    "进行",
    "通过",
    "不同",
    "其中",
}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}:{digest}"


def _provenance(document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "chunk_id": document_key(document),
        "source_file": metadata.get("source_file"),
        "heading_path": metadata.get("heading_path") or metadata.get("section"),
        "page": metadata.get("page") or metadata.get("page_number"),
        "parent_id": metadata.get("parent_id"),
    }


@dataclass(frozen=True)
class ExtractedEntity:
    """保存规范化实体及其抽取置信度。"""

    name: str
    normalized: str
    confidence: float
    source: str


@dataclass
class KnowledgeGraphExtractor:
    """在无外部模型条件下抽取可审计图谱。"""

    max_entities_per_document: int = 12
    max_claims_per_document: int = 4
    use_shared_terms: bool = True
    use_entities: bool = True
    use_claims: bool = True

    def process(self, documents: Iterable[Document]) -> KnowledgeGraph:
        docs = list(documents)
        graph = KnowledgeGraph()
        evidence_ids: list[str] = []
        document_entities: dict[str, list[str]] = {}
        for document in docs:
            evidence_id = self._add_evidence(graph, document)
            evidence_ids.append(evidence_id)
            entity_ids = self._add_entities(graph, document, evidence_id) if self.use_entities else []
            document_entities[evidence_id] = entity_ids
            if self.use_claims:
                self._add_claims(graph, document, evidence_id, entity_ids)
        self._add_structural_relations(graph, docs, evidence_ids)
        if self.use_entities:
            self._add_entity_co_occurrence(graph, document_entities)
        graph.validate()
        return graph

    @staticmethod
    def _add_evidence(graph: KnowledgeGraph, document: Document) -> str:
        chunk_id = document_key(document)
        node_id = f"evidence:{chunk_id}"
        metadata = dict(document.metadata or {})
        graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=GraphNodeType.EVIDENCE,
                label=str(metadata.get("heading_path") or metadata.get("section") or chunk_id),
                properties={
                    **_provenance(document),
                    "text_preview": (document.page_content or "")[:240],
                    "is_parent": bool(metadata.get("is_parent", False)),
                },
            )
        )
        return node_id

    def _add_entities(self, graph: KnowledgeGraph, document: Document, evidence_id: str) -> list[str]:
        entities = self.extract_entities(document)
        entity_ids: list[str] = []
        for entity in entities:
            entity_id = _stable_id("entity", entity.normalized)
            graph.add_node(
                GraphNode(
                    node_id=entity_id,
                    node_type=GraphNodeType.ENTITY,
                    label=entity.name,
                    properties={"normalized": entity.normalized},
                )
            )
            graph.add_relation(
                GraphRelation(
                    source=evidence_id,
                    target=entity_id,
                    relation_type=GraphRelationType.MENTIONS,
                    weight=1.0,
                    confidence=entity.confidence,
                    provenance={**_provenance(document), "extractor": entity.source},
                )
            )
            entity_ids.append(entity_id)
        return entity_ids

    def _add_claims(
        self,
        graph: KnowledgeGraph,
        document: Document,
        evidence_id: str,
        entity_ids: list[str],
    ) -> None:
        claims = self.extract_claims(document.page_content or "")
        entity_labels = {node_id: graph.nodes[node_id].label.casefold() for node_id in entity_ids}
        for claim in claims:
            claim_id = _stable_id("claim", f"{document_key(document)}|{claim}")
            graph.add_node(
                GraphNode(
                    node_id=claim_id,
                    node_type=GraphNodeType.CLAIM,
                    label=claim,
                    properties={**_provenance(document), "text": claim},
                )
            )
            graph.add_relation(
                GraphRelation(
                    source=evidence_id,
                    target=claim_id,
                    relation_type=GraphRelationType.SUPPORTS,
                    weight=1.0,
                    confidence=0.8,
                    provenance={**_provenance(document), "extractor": "rule_claim_v1"},
                )
            )
            lowered = claim.casefold()
            for entity_id, label in entity_labels.items():
                if label and label in lowered:
                    graph.add_relation(
                        GraphRelation(
                            source=claim_id,
                            target=entity_id,
                            relation_type=GraphRelationType.ABOUT,
                            weight=0.7,
                            confidence=0.8,
                            provenance={**_provenance(document), "extractor": "claim_entity_match"},
                        )
                    )

    def _add_structural_relations(
        self,
        graph: KnowledgeGraph,
        documents: list[Document],
        evidence_ids: list[str],
    ) -> None:
        terms = [self._terms(document.page_content or "") for document in documents]
        for left_index, right_index in combinations(range(len(documents)), 2):
            left = documents[left_index]
            right = documents[right_index]
            left_meta = left.metadata or {}
            right_meta = right.metadata or {}
            provenance = {
                "left_chunk_id": document_key(left),
                "right_chunk_id": document_key(right),
                "extractor": "structural_v1",
            }
            candidates: list[tuple[GraphRelationType, float, dict[str, Any]]] = []
            if left_meta.get("parent_id") and left_meta.get("parent_id") == right_meta.get("parent_id"):
                candidates.append((GraphRelationType.SAME_PARENT, 3.0, provenance))
            if left_meta.get("source_file") and left_meta.get("source_file") == right_meta.get("source_file"):
                candidates.append((GraphRelationType.SAME_SOURCE, 1.0, provenance))
            left_heading = left_meta.get("heading_path") or left_meta.get("section")
            right_heading = right_meta.get("heading_path") or right_meta.get("section")
            if left_heading and left_heading == right_heading:
                candidates.append((GraphRelationType.SAME_SECTION, 1.0, provenance))
            overlap = terms[left_index] & terms[right_index]
            if self.use_shared_terms and overlap:
                candidates.append(
                    (
                        GraphRelationType.SHARED_TERMS,
                        min(2.0, len(overlap) / 3.0),
                        {**provenance, "terms": sorted(overlap)[:12]},
                    )
                )
            for relation_type, weight, relation_provenance in candidates:
                self._add_bidirectional_relation(
                    graph,
                    evidence_ids[left_index],
                    evidence_ids[right_index],
                    relation_type,
                    weight,
                    relation_provenance,
                )

    @staticmethod
    def _add_bidirectional_relation(
        graph: KnowledgeGraph,
        left: str,
        right: str,
        relation_type: GraphRelationType,
        weight: float,
        provenance: dict[str, Any],
    ) -> None:
        graph.add_relation(
            GraphRelation(
                source=left,
                target=right,
                relation_type=relation_type,
                weight=weight,
                confidence=1.0,
                provenance=provenance,
            )
        )
        graph.add_relation(
            GraphRelation(
                source=right,
                target=left,
                relation_type=relation_type,
                weight=weight,
                confidence=1.0,
                provenance=provenance,
            )
        )

    @staticmethod
    def _add_entity_co_occurrence(graph: KnowledgeGraph, document_entities: dict[str, list[str]]) -> None:
        for evidence_id, entity_ids in document_entities.items():
            for left, right in combinations(sorted(set(entity_ids)), 2):
                provenance = {"evidence_id": evidence_id, "extractor": "entity_co_occurrence_v1"}
                KnowledgeGraphExtractor._add_bidirectional_relation(
                    graph,
                    left,
                    right,
                    GraphRelationType.CO_OCCURS,
                    0.5,
                    provenance,
                )

    def extract_entities(self, document: Document) -> list[ExtractedEntity]:
        metadata = document.metadata or {}
        extracted: dict[str, ExtractedEntity] = {}
        raw_entities = metadata.get("entities") or []
        if isinstance(raw_entities, (str, dict)):
            raw_entities = [raw_entities]
        if isinstance(raw_entities, list):
            for raw in raw_entities:
                if isinstance(raw, dict):
                    name = str(raw.get("name") or raw.get("text") or "").strip()
                    confidence = float(raw.get("confidence", 1.0))
                else:
                    name = str(raw).strip()
                    confidence = 1.0
                self._store_entity(extracted, name, confidence, "metadata")
        text = document.page_content or ""
        for name in _ENGLISH_ENTITY_RE.findall(text):
            self._store_entity(extracted, name, 0.75, "rule_english")
        frequencies: dict[str, int] = {}
        for term in _CHINESE_TERM_RE.findall(text):
            if term in _BLOCKED_TERMS or len(term) > 8:
                continue
            frequencies[term] = frequencies.get(term, 0) + 1
        for name, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0])):
            self._store_entity(extracted, name, 0.55, "rule_chinese")
        return sorted(
            extracted.values(),
            key=lambda entity: (-entity.confidence, entity.normalized),
        )[: self.max_entities_per_document]

    def extract_claims(self, text: str) -> list[str]:
        claims: list[str] = []
        for sentence in _SENTENCE_RE.split(text):
            normalized = " ".join(sentence.split()).strip()
            if 12 <= len(normalized) <= 320 and _CLAIM_MARKERS.search(normalized):
                claims.append(normalized)
        return claims[: self.max_claims_per_document]

    @staticmethod
    def _store_entity(
        output: dict[str, ExtractedEntity],
        name: str,
        confidence: float,
        source: str,
    ) -> None:
        clean = " ".join(name.split()).strip(".,;:()[]{}")
        normalized = clean.casefold()
        if not clean or normalized in _BLOCKED_TERMS or len(clean) < 2:
            return
        candidate = ExtractedEntity(
            name=clean,
            normalized=normalized,
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
        )
        existing = output.get(normalized)
        if existing is None or candidate.confidence > existing.confidence:
            output[normalized] = candidate

    @staticmethod
    def _terms(text: str) -> set[str]:
        english = re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
        return {term for term in english + chinese if term not in _BLOCKED_TERMS}
