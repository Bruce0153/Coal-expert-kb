"""定义评估数据、运行观察和结果模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class QueryType(StrEnum):
    """评估问题类型。"""

    FACT = "fact"
    CONDITION = "condition"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class EvidenceReference:
    """保存可追溯到原始文档的证据标注。"""

    source_file: str | None = None
    document_id: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    text_span: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceReference:
        return cls(
            source_file=value.get("source_file") or value.get("source_contains"),
            document_id=value.get("document_id"),
            page=value.get("page"),
            section=value.get("section"),
            chunk_id=value.get("chunk_id"),
            text_span=value.get("text_span"),
        )


@dataclass(frozen=True)
class EvaluationCase:
    """表示一条版本化评估样本。"""

    case_id: str
    query: str
    query_type: QueryType = QueryType.FACT
    expected_answer: str | None = None
    expected_evidence: tuple[EvidenceReference, ...] = ()
    expected_filters: dict[str, Any] = field(default_factory=dict)
    answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, row_number: int) -> EvaluationCase:
        evidence_values = value.get("expected_evidence") or value.get("gold_sources") or []
        case_id = str(value.get("id") or value.get("case_id") or f"case_{row_number:05d}")
        query = str(value.get("query") or value.get("question") or "").strip()
        if not query:
            raise ValueError(f"Evaluation case {case_id} has an empty query")
        query_type = QueryType(str(value.get("query_type") or QueryType.FACT.value))
        return cls(
            case_id=case_id,
            query=query,
            query_type=query_type,
            expected_answer=value.get("expected_answer"),
            expected_evidence=tuple(EvidenceReference.from_dict(item) for item in evidence_values),
            expected_filters=dict(value.get("expected_filters") or {}),
            answerable=bool(value.get("answerable", True)),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("case_id")
        payload["query_type"] = self.query_type.value
        payload["expected_evidence"] = list(payload["expected_evidence"])
        return payload


@dataclass(frozen=True)
class RetrievedEvidence:
    """保存一个排序后的检索结果。"""

    rank: int
    source_file: str | None
    document_id: str | None
    page: int | None
    section: str | None
    chunk_id: str | None
    text: str
    score: float | None = None

    @classmethod
    def from_document(cls, document: Any, *, rank: int) -> RetrievedEvidence:
        metadata = document.metadata or {}
        return cls(
            rank=rank,
            source_file=metadata.get("source_file"),
            document_id=metadata.get("document_id"),
            page=metadata.get("page"),
            section=metadata.get("section") or metadata.get("heading_path"),
            chunk_id=metadata.get("chunk_id"),
            text=document.page_content or "",
            score=metadata.get("score") or metadata.get("retrieval_score"),
        )


@dataclass(frozen=True)
class ClaimObservation:
    """保存回答中的一个 Claim 及其证据状态。"""

    text: str
    citations: tuple[str, ...] = ()
    supported: bool = False


@dataclass(frozen=True)
class AnswerObservation:
    """保存回答评估所需的结构化输出。"""

    answer_text: str
    citations: tuple[EvidenceReference, ...] = ()
    claims: tuple[ClaimObservation, ...] = ()
    abstained: bool = False


@dataclass(frozen=True)
class EvaluationObservation:
    """保存一次案例执行产生的检索与回答观察。"""

    retrieved: tuple[RetrievedEvidence, ...]
    answer: AnswerObservation | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class CaseEvaluationResult:
    """保存一条案例的全部指标和失败归因。"""

    case_id: str
    query: str
    query_type: str
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]
    failure_category: str
    latency_ms: float
    retrieved: tuple[RetrievedEvidence, ...]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
