"""构建 Milestone C 的复杂科学问答核心模型、规划器和执行器。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"未找到替换目标: {path}: {old[:80]}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def process() -> None:
    _write(
        "src/coal_kb/core/models/query.py",
        '''"""定义查询规划、复杂问答路线、检索策略与回答约束。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal[
    "fact",
    "comparison",
    "multi_hop",
    "aggregation",
    "table",
    "cross_document",
    "unanswerable",
]
AggregationOperation = Literal["count", "sum", "average", "min", "max", "median", "group_by", "top_k"]


class Constraint(BaseModel):
    """表示查询中的一个硬约束或软约束。"""

    field: str
    op: str = "eq"
    value: Any
    priority: Literal["hard", "soft"] = "soft"
    confidence: float = 0.5
    source: str = "rule"
    note: Optional[str] = None


class QueryUnderstanding(BaseModel):
    """保存原始问题、标准化问题和领域约束。"""

    raw: str
    normalized: str
    language: str = "zh"
    rewritten: Optional[str] = None
    rewrite_reason: Optional[str] = None
    hard_constraints: List[Constraint] = Field(default_factory=list)
    soft_constraints: List[Constraint] = Field(default_factory=list)


class SubQuerySpec(BaseModel):
    """表示复杂问题中的一个可回放子问题。"""

    subquery_id: str
    query: str
    purpose: str
    depends_on: List[str] = Field(default_factory=list)


class AggregationSpec(BaseModel):
    """表示结构化记录上的确定性聚合请求。"""

    operation: AggregationOperation = "count"
    field: Optional[str] = None
    group_by: Optional[str] = None
    top_k: int = 10


class ComplexQuestionSpec(BaseModel):
    """保存复杂科学问答的统一路由结果。"""

    query_type: QuestionType = "fact"
    confidence: float = 1.0
    reason: str = "默认事实检索"
    subqueries: List[SubQuerySpec] = Field(default_factory=list)
    comparison_entities: List[str] = Field(default_factory=list)
    aggregation: Optional[AggregationSpec] = None
    require_table: bool = False
    require_cross_document: bool = False


class RetrievalStep(BaseModel):
    """表示一个检索阶段。"""

    name: str
    level: Literal["parent", "child", "single"]
    fusion_mode: Literal["vector", "bm25", "rrf"] = "rrf"
    k_candidates: int
    k_final: int
    where_mode: Literal["hard_only", "full"] = "hard_only"
    enable_relax: bool = False


class RelaxRule(BaseModel):
    """表示一次约束放宽规则。"""

    drop_fields: List[str] = Field(default_factory=list)
    widen_ranges: Dict[str, float] = Field(default_factory=dict)
    soften_priority: bool = True


class RelaxPolicy(BaseModel):
    """定义最多允许执行的约束放宽步骤。"""

    max_steps: int = 2
    rules: List[RelaxRule] = Field(default_factory=list)


class RerankSpec(BaseModel):
    """定义重排序开关和输出数量。"""

    enabled: bool = False
    top_n: int = 10


class NeighborSpec(BaseModel):
    """定义相邻证据扩展。"""

    enabled: bool = False
    window: int = 1


class DiversitySpec(BaseModel):
    """定义单一来源的最大证据数量。"""

    max_per_source: int = 2


class ContextSpec(BaseModel):
    """定义上下文预算和去重方式。"""

    max_context_tokens: int = 1800
    max_evidence_chunks: int = 8
    group_by_heading: bool = True
    deduplicate: bool = True
    dedup_mode: Literal["text", "embedding", "off"] = "text"


class AnswerSpec(BaseModel):
    """定义回答引用和拒答要求。"""

    require_citations: bool = True
    refuse_threshold: float = 0.3
    min_evidence: int = 2
    output_format: Literal["markdown", "json"] = "markdown"


class ObservabilitySpec(BaseModel):
    """定义 Trace 和调试配置。"""

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    log_plan: bool = True
    debug: bool = False


class QueryPlan(BaseModel):
    """统一表示事实检索和复杂科学问答计划。"""

    query: QueryUnderstanding
    complex: ComplexQuestionSpec = Field(default_factory=ComplexQuestionSpec)
    retrieval_steps: List[RetrievalStep] = Field(default_factory=list)
    relax_policy: RelaxPolicy = Field(default_factory=RelaxPolicy)
    rerank: RerankSpec = Field(default_factory=RerankSpec)
    neighbor: NeighborSpec = Field(default_factory=NeighborSpec)
    diversity: DiversitySpec = Field(default_factory=DiversitySpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    answer: AnswerSpec = Field(default_factory=AnswerSpec)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
''',
    )

    _write(
        "src/coal_kb/complex_qa/config.py",
        '''"""维护复杂科学问答实际使用的确定性参数。"""

# 路由与分解
MAX_SUBQUERIES = 4
MAX_MULTI_HOP_STEPS = 3

# 检索预算
COMPARISON_K_PER_SIDE = 4
CROSS_DOCUMENT_MIN_SOURCES = 2
CROSS_DOCUMENT_MAX_PER_SOURCE = 2
TABLE_TOP_K = 5
AGGREGATION_RECORD_LIMIT = 500
AGGREGATION_EVIDENCE_LIMIT = 12

# 数据路径
TABLE_RECORDS_PATH = "data/interim/table_records.jsonl"

# 可复现性
SAMPLE_SEED = 20260715
''',
    )

    _write(
        "src/coal_kb/complex_qa/models.py",
        '''"""定义复杂科学问答执行结果、聚合结果和标准表格记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ComplexExecutionResult:
    """保存复杂路线产生的证据和可回放 Trace。"""

    documents: list[Document]
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationResult:
    """保存程序计算的聚合结果和参与记录。"""

    operation: str
    field: str | None
    value: Any
    sample_size: int
    records: tuple[dict[str, Any], ...]


class TableRecord(BaseModel):
    """表示从科学文档表格中恢复的结构化内容。"""

    table_id: str
    source_file: str
    page: int | None = None
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    nearby_text: str = ""
''',
    )

    _write(
        "src/coal_kb/complex_qa/router.py",
        '''"""使用可解释规则识别复杂科学问答路线。"""

from __future__ import annotations

import re

from coal_kb.core.models.query import QuestionType


def route_question(query: str) -> tuple[QuestionType, float, str]:
    """返回问题类型、置信度和可解释原因。"""
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return "unanswerable", 1.0, "问题为空"

    if re.search(r"未公开|无公开记录|无法从文献确认|不存在的实验|undocumented|not published", normalized):
        return "unanswerable", 0.95, "问题明确要求无法由知识库证实的信息"

    if re.search(r"表中|表格|第\s*\d+\s*行|哪一列|单元格|table\s*\d+", normalized):
        return "table", 0.95, "问题明确引用表格、行、列或单元格"

    if re.search(r"平均|均值|中位数|最高|最低|最大|最小|前\s*\d+|top\s*\d+|排名|频率|出现最多|共有多少|多少篇|多少条|统计", normalized):
        return "aggregation", 0.92, "问题要求确定性统计、排序或聚合"

    if re.search(r"多篇文献|不同研究|文献共识|总体结论|综合来看|研究结论是否一致|冲突结论|跨文档|综述", normalized):
        return "cross_document", 0.91, "问题要求跨文档共识、差异或冲突综合"

    if re.search(r"比较|对比|区别|差异|相比| versus |\bvs\.?\b|与.+有何不同|和.+有何不同", f" {normalized} "):
        return "comparison", 0.9, "问题包含明确比较关系"

    if re.search(r"为什么|如何通过|怎样导致|共同导致|反应路径|机制链|从而|进而|因果链|multi[- ]hop", normalized):
        return "multi_hop", 0.84, "问题需要连接中间过程或因果证据"

    return "fact", 0.8, "未命中复杂路线规则，使用事实检索"
''',
    )

    _write(
        "src/coal_kb/complex_qa/planning.py",
        '''"""把路由结果转换为可执行子问题和结构化操作。"""

from __future__ import annotations

import re

from coal_kb.core.models.query import AggregationSpec, ComplexQuestionSpec, SubQuerySpec
from coal_kb.complex_qa.router import route_question

_FIELD_PATTERNS = (
    (r"温度|temperature", "T_K"),
    (r"压力|pressure", "P_MPa"),
    (r"nh3|氨", "pollutants.NH3"),
    (r"hcn|氰化氢", "pollutants.HCN"),
    (r"h2|氢气|氢产率", "pollutants.H2"),
    (r"co2|二氧化碳", "pollutants.CO2"),
    (r"co(?!2)|一氧化碳", "pollutants.CO"),
    (r"ch4|甲烷", "pollutants.CH4"),
    (r"焦油|tar", "pollutants.tar"),
)
_GROUP_PATTERNS = (
    (r"按煤种|不同煤种", "coal_name"),
    (r"按阶段|不同阶段", "stage"),
    (r"按反应器|不同反应器", "reactor_type"),
    (r"按气化剂|不同气化剂", "gas_agent"),
)


def _comparison_entities(query: str) -> list[str]:
    patterns = (
        r"比较\s*(.+?)\s*(?:与|和|及|vs\.?|versus)\s*(.+?)(?:对|在|的|之间|有何|$)",
        r"(.+?)\s*(?:与|和|vs\.?)\s*(.+?)\s*(?:有何不同|的区别|的差异)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.I)
        if match:
            values = [" ".join(value.strip(" ，。；;：:").split()) for value in match.groups()[:2]]
            return [value for value in values if value]
    return []


def _aggregation_spec(query: str) -> AggregationSpec:
    lowered = query.lower()
    operation = "count"
    if re.search(r"平均|均值|average|mean", lowered):
        operation = "average"
    elif re.search(r"中位数|median", lowered):
        operation = "median"
    elif re.search(r"最高|最大|max", lowered):
        operation = "max"
    elif re.search(r"最低|最小|min", lowered):
        operation = "min"
    elif re.search(r"前\s*\d+|top\s*\d+|排名", lowered):
        operation = "top_k"
    elif re.search(r"按.+统计|分组|group", lowered):
        operation = "group_by"

    field = None
    for pattern, canonical in _FIELD_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            field = canonical
            break

    group_by = None
    for pattern, canonical in _GROUP_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            group_by = canonical
            break
    if group_by and operation == "count":
        operation = "group_by"

    top_match = re.search(r"(?:前|top)\s*(\d+)", lowered, flags=re.I)
    top_k = max(1, min(100, int(top_match.group(1)))) if top_match else 10
    return AggregationSpec(operation=operation, field=field, group_by=group_by, top_k=top_k)


def build_complex_spec(query: str, *, max_subqueries: int, max_multi_hop_steps: int) -> ComplexQuestionSpec:
    """根据问题生成统一复杂问答计划。"""
    query_type, confidence, reason = route_question(query)
    subqueries: list[SubQuerySpec] = []
    entities: list[str] = []
    aggregation = None

    if query_type == "comparison":
        entities = _comparison_entities(query)
        if len(entities) == 2:
            for index, entity in enumerate(entities, start=1):
                subqueries.append(
                    SubQuerySpec(
                        subquery_id=f"comparison_{index}",
                        query=f"{entity} 实验条件 结果 反应机理",
                        purpose=f"检索比较对象 {entity} 的独立证据",
                    )
                )
        else:
            subqueries = [
                SubQuerySpec(subquery_id="comparison_1", query=f"{query} 对象一 证据", purpose="检索第一侧证据"),
                SubQuerySpec(subquery_id="comparison_2", query=f"{query} 对象二 证据", purpose="检索第二侧证据"),
            ]
    elif query_type == "multi_hop":
        templates = (
            ("hop_1", f"{query} 关键反应 中间过程", "识别关键中间过程", []),
            ("hop_2", f"{query} 中间过程 对目标结果的影响", "连接中间过程与目标结果", ["hop_1"]),
            ("hop_3", f"{query} 实验条件 机制链 证据", "核验完整机制链及适用条件", ["hop_2"]),
        )
        for subquery_id, subquery, purpose, dependencies in templates[:max_multi_hop_steps]:
            subqueries.append(SubQuerySpec(subquery_id=subquery_id, query=subquery, purpose=purpose, depends_on=dependencies))
    elif query_type == "aggregation":
        aggregation = _aggregation_spec(query)
    elif query_type == "table":
        subqueries.append(SubQuerySpec(subquery_id="table_1", query=query, purpose="定位相关表格、行和单元格"))
    elif query_type == "cross_document":
        templates = (
            ("cross_support", f"{query} 支持性证据", "检索支持主要结论的文档"),
            ("cross_conflict", f"{query} 相反结果 冲突证据", "检索相反或冲突结论"),
            ("cross_conditions", f"{query} 实验条件差异", "解释文献差异的条件来源"),
        )
        subqueries = [SubQuerySpec(subquery_id=item[0], query=item[1], purpose=item[2]) for item in templates]

    return ComplexQuestionSpec(
        query_type=query_type,
        confidence=confidence,
        reason=reason,
        subqueries=subqueries[:max_subqueries],
        comparison_entities=entities,
        aggregation=aggregation,
        require_table=query_type == "table",
        require_cross_document=query_type == "cross_document",
    )
''',
    )

    _write(
        "src/coal_kb/complex_qa/utils/__init__.py",
        '''"""导出复杂科学问答的文档处理公共函数。"""

from coal_kb.complex_qa.utils.documents import clone_plan_for_subquery, deduplicate_documents, tag_documents

__all__ = ["clone_plan_for_subquery", "deduplicate_documents", "tag_documents"]
''',
    )

    _write(
        "src/coal_kb/complex_qa/utils/documents.py",
        '''"""提供复杂路线复用的 QueryPlan 克隆、文档标记和去重函数。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.core.models.query import ComplexQuestionSpec, QueryPlan, SubQuerySpec


def clone_plan_for_subquery(plan: QueryPlan, subquery: SubQuerySpec) -> QueryPlan:
    """克隆计划并把复杂子问题降级为一次普通事实检索。"""
    cloned = plan.model_copy(deep=True)
    cloned.query.raw = subquery.query
    cloned.query.normalized = subquery.query
    cloned.query.rewritten = None
    cloned.complex = ComplexQuestionSpec(query_type="fact", confidence=1.0, reason="复杂路线内部子检索")
    return cloned


def tag_documents(documents: list[Document], **metadata: object) -> list[Document]:
    """复制文档并附加复杂路线元数据，避免修改召回缓存对象。"""
    tagged: list[Document] = []
    for document in documents:
        merged = dict(document.metadata or {})
        merged.update(metadata)
        tagged.append(Document(page_content=document.page_content, metadata=merged))
    return tagged


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """按 chunk_id 或来源页码稳定去重。"""
    seen: set[str] = set()
    output: list[Document] = []
    for document in documents:
        metadata = document.metadata or {}
        key = str(metadata.get("chunk_id") or f"{metadata.get('source_file')}|{metadata.get('page')}|{document.page_content[:80]}")
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output
''',
    )

    _write(
        "src/coal_kb/complex_qa/comparison.py",
        '''"""执行比较问题的双侧独立检索和证据标记。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.utils import clone_plan_for_subquery, deduplicate_documents, tag_documents
from coal_kb.core.models.query import QueryPlan


@dataclass
class ComparisonExecutor:
    """持有检索器并执行比较问题。"""

    retriever: Any
    k_per_side: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        documents = []
        steps = []
        for index, subquery in enumerate(plan.complex.subqueries[:2]):
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(clone_plan_for_subquery(plan, subquery), trace=local_trace)[: self.k_per_side]
            entity = plan.complex.comparison_entities[index] if index < len(plan.complex.comparison_entities) else subquery.subquery_id
            documents.extend(
                tag_documents(
                    hits,
                    complex_route="comparison",
                    complex_role=subquery.subquery_id,
                    comparison_entity=entity,
                )
            )
            steps.append({"subquery_id": subquery.subquery_id, "query": subquery.query, "entity": entity, "hits": len(hits)})
        return ComplexExecutionResult(
            documents=deduplicate_documents(documents),
            trace={"query_type": "comparison", "steps": steps, "entities": plan.complex.comparison_entities},
        )
''',
    )

    _write(
        "src/coal_kb/complex_qa/multi_hop.py",
        '''"""执行受限多跳检索并保存每一跳的查询和证据。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.utils import clone_plan_for_subquery, deduplicate_documents, tag_documents
from coal_kb.core.models.query import QueryPlan


@dataclass
class MultiHopExecutor:
    """持有检索器并顺序执行有依赖关系的子问题。"""

    retriever: Any
    max_steps: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        documents = []
        steps = []
        bridge_terms: list[str] = []
        for subquery in plan.complex.subqueries[: self.max_steps]:
            effective_query = subquery.query
            if subquery.depends_on and bridge_terms:
                effective_query = f"{effective_query} {' '.join(bridge_terms)}"
            effective = subquery.model_copy(update={"query": effective_query})
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(clone_plan_for_subquery(plan, effective), trace=local_trace)
            tagged = tag_documents(hits, complex_route="multi_hop", complex_role=subquery.subquery_id)
            documents.extend(tagged)
            bridge_terms = self._extract_bridge_terms(hits)
            steps.append(
                {
                    "subquery_id": subquery.subquery_id,
                    "query": effective_query,
                    "depends_on": subquery.depends_on,
                    "bridge_terms": bridge_terms,
                    "hits": len(hits),
                }
            )
        return ComplexExecutionResult(
            documents=deduplicate_documents(documents),
            trace={"query_type": "multi_hop", "steps": steps, "chain_complete": bool(steps) and all(step["hits"] > 0 for step in steps)},
        )

    @staticmethod
    def _extract_bridge_terms(documents: list[Any]) -> list[str]:
        """从上一跳证据中稳定抽取少量中间术语。"""
        text = " ".join(str(document.page_content or "")[:800] for document in documents[:3]).lower()
        terms = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,6}", text)
        blocked = {"研究", "结果", "实验", "条件", "影响", "过程", "表明", "通过", "进行"}
        counts = Counter(term for term in terms if term not in blocked)
        return [term for term, _ in counts.most_common(4)]
''',
    )

    _write(
        "src/coal_kb/complex_qa/aggregation.py",
        '''"""在结构化实验记录上执行可复现统计并生成证据文档。"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.complex_qa.models import AggregationResult, ComplexExecutionResult
from coal_kb.core.models.query import Constraint, QueryPlan


@dataclass
class AggregationRepository:
    """从现有 SQLite 记录库读取结构化实验数据。"""

    sqlite_path: str
    record_limit: int

    def load_records(self) -> list[dict[str, Any]]:
        path = Path(self.sqlite_path)
        if not path.exists():
            return []
        query = """
            SELECT r.*, e.page, e.chunk_id, e.quote
            FROM records AS r
            LEFT JOIN evidence AS e ON e.record_id = r.record_id
            ORDER BY r.updated_at DESC
            LIMIT ?
        """
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute(query, (self.record_limit,)).fetchall()
            except sqlite3.DatabaseError:
                return []
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            payload = dict(row)
            record_id = str(payload.get("record_id") or "")
            if record_id in seen:
                continue
            seen.add(record_id)
            for field in ("gas_agent_json", "ratios_json", "pollutants_json"):
                value = payload.pop(field, None)
                canonical = field.removesuffix("_json")
                try:
                    payload[canonical] = json.loads(value) if value else ([] if canonical == "gas_agent" else {})
                except json.JSONDecodeError:
                    payload[canonical] = [] if canonical == "gas_agent" else {}
            records.append(payload)
        return records


@dataclass
class AggregationExecutor:
    """执行过滤、聚合和证据文档构建。"""

    repository: AggregationRepository
    retriever: Any
    evidence_limit: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        records = self.repository.load_records()
        filtered = [record for record in records if self._matches_constraints(record, plan.query.hard_constraints + plan.query.soft_constraints)]
        spec = plan.complex.aggregation
        if spec is None or not filtered:
            fallback = self.retriever.execute(plan, trace={})
            return ComplexExecutionResult(
                documents=fallback,
                trace={"query_type": "aggregation", "fallback": "document_retrieval", "record_count": len(filtered)},
            )

        result = self._aggregate(filtered, spec.operation, spec.field, spec.group_by, spec.top_k)
        documents = [
            Document(
                page_content=(
                    f"程序计算结果：operation={result.operation}; field={result.field}; "
                    f"sample_size={result.sample_size}; value={json.dumps(result.value, ensure_ascii=False)}"
                ),
                metadata={
                    "source_file": "computed_aggregation",
                    "section": "aggregation",
                    "chunk_id": f"aggregation-{plan.observability.trace_id}",
                    "record_ids": [record.get("record_id") for record in result.records],
                },
            )
        ]
        for record in result.records[: self.evidence_limit]:
            documents.append(self._record_document(record))
        return ComplexExecutionResult(
            documents=documents,
            trace={
                "query_type": "aggregation",
                "operation": result.operation,
                "field": result.field,
                "sample_size": result.sample_size,
                "value": result.value,
                "record_ids": [record.get("record_id") for record in result.records],
            },
        )

    def _aggregate(self, records: list[dict[str, Any]], operation: str, field: str | None, group_by: str | None, top_k: int) -> AggregationResult:
        if operation == "count" and not field:
            return AggregationResult(operation=operation, field=field, value=len(records), sample_size=len(records), records=tuple(records))
        if operation == "group_by":
            key = group_by or field or "stage"
            groups: dict[str, int] = {}
            for record in records:
                value = self._value(record, key)
                label = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value or "unknown")
                groups[label] = groups.get(label, 0) + 1
            return AggregationResult(operation=operation, field=key, value=groups, sample_size=len(records), records=tuple(records))

        numeric = [(record, self._numeric_value(record, field)) for record in records]
        numeric = [(record, value) for record, value in numeric if value is not None]
        if not numeric:
            return AggregationResult(operation=operation, field=field, value=None, sample_size=0, records=())
        values = [value for _, value in numeric]
        selected_records = [record for record, _ in numeric]
        value: Any
        if operation == "sum":
            value = sum(values)
        elif operation == "average":
            value = statistics.fmean(values)
        elif operation == "median":
            value = statistics.median(values)
        elif operation == "min":
            value = min(values)
        elif operation == "max":
            value = max(values)
        elif operation == "top_k":
            ranked = sorted(numeric, key=lambda item: item[1], reverse=True)[:top_k]
            value = [{"record_id": record.get("record_id"), "value": score} for record, score in ranked]
            selected_records = [record for record, _ in ranked]
        else:
            value = len(values)
        return AggregationResult(operation=operation, field=field, value=value, sample_size=len(values), records=tuple(selected_records))

    @staticmethod
    def _matches_constraints(record: dict[str, Any], constraints: list[Constraint]) -> bool:
        for constraint in constraints:
            if constraint.priority != "hard":
                continue
            if constraint.op == "range" and isinstance(constraint.value, list) and len(constraint.value) == 2:
                field = "T_K" if constraint.field == "T_range_K" else "P_MPa"
                value = record.get(field)
                if value is None or not float(constraint.value[0]) <= float(value) <= float(constraint.value[1]):
                    return False
            elif constraint.op in {"enum", "eq"}:
                if str(record.get(constraint.field, "")).lower() != str(constraint.value).lower():
                    return False
            elif constraint.op == "set":
                actual = record.get(constraint.field) or []
                expected = constraint.value if isinstance(constraint.value, list) else [constraint.value]
                if not set(map(str, expected)) & set(map(str, actual if isinstance(actual, list) else [actual])):
                    return False
        return True

    @staticmethod
    def _value(record: dict[str, Any], field: str | None) -> Any:
        if not field:
            return None
        if field.startswith("pollutants."):
            key = field.split(".", 1)[1]
            pollutants = record.get("pollutants") or {}
            value = pollutants.get(key)
            if value is None:
                value = next((item for name, item in pollutants.items() if str(name).lower() == key.lower()), None)
            return value
        return record.get(field)

    def _numeric_value(self, record: dict[str, Any], field: str | None) -> float | None:
        value = self._value(record, field)
        if isinstance(value, dict):
            value = value.get("value_norm", value.get("value"))
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _record_document(record: dict[str, Any]) -> Document:
        content = {
            "record_id": record.get("record_id"),
            "stage": record.get("stage"),
            "coal_name": record.get("coal_name"),
            "reactor_type": record.get("reactor_type"),
            "T_K": record.get("T_K"),
            "P_MPa": record.get("P_MPa"),
            "gas_agent": record.get("gas_agent"),
            "pollutants": record.get("pollutants"),
        }
        return Document(
            page_content=f"结构化实验记录：{json.dumps(content, ensure_ascii=False, sort_keys=True)}",
            metadata={
                "source_file": record.get("source_file") or "structured_record",
                "page": record.get("page"),
                "chunk_id": record.get("chunk_id") or record.get("record_id"),
                "section": "structured_record",
                "record_id": record.get("record_id"),
            },
        )
''',
    )

    _write(
        "src/coal_kb/complex_qa/tables.py",
        '''"""读取标准表格记录并执行表格、行和单元格检索。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.complex_qa.models import ComplexExecutionResult, TableRecord
from coal_kb.core.models.query import QueryPlan


@dataclass
class TableRepository:
    """从 JSONL 表格资产中加载并检索记录。"""

    records_path: str

    def load(self) -> list[TableRecord]:
        path = Path(self.records_path)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        records: list[TableRecord] = []
        for line in lines:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(TableRecord.model_validate(payload))
        return records

    def search(self, query: str, *, top_k: int) -> list[tuple[TableRecord, dict[str, Any], float]]:
        query_terms = self._terms(query)
        candidates: list[tuple[TableRecord, dict[str, Any], float]] = []
        for table in self.load():
            for row in table.rows:
                searchable = " ".join([table.caption, " ".join(table.headers), table.nearby_text, json.dumps(row, ensure_ascii=False)])
                terms = self._terms(searchable)
                score = len(query_terms & terms) / max(1, len(query_terms))
                if score > 0:
                    candidates.append((table, row, score))
        return sorted(candidates, key=lambda item: (-item[2], item[0].table_id))[:top_k]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_.%+-]+|[\u4e00-\u9fff]", text.lower()))


@dataclass
class TableExecutor:
    """执行表格路线并在没有表格资产时安全回退到文档检索。"""

    repository: TableRepository
    retriever: Any
    top_k: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        matches = self.repository.search(plan.query.normalized, top_k=self.top_k)
        if not matches:
            fallback = self.retriever.execute(plan, trace={})
            table_docs = [document for document in fallback if "table" in str((document.metadata or {}).get("section", "")).lower()]
            return ComplexExecutionResult(
                documents=table_docs or fallback,
                trace={"query_type": "table", "fallback": "document_retrieval", "table_matches": 0},
            )
        documents = []
        for table, row, score in matches:
            documents.append(
                Document(
                    page_content=(
                        f"表格标题：{table.caption}\n"
                        f"表头：{json.dumps(table.headers, ensure_ascii=False)}\n"
                        f"命中行：{json.dumps(row, ensure_ascii=False, sort_keys=True)}"
                    ),
                    metadata={
                        "source_file": table.source_file,
                        "page": table.page,
                        "section": "table",
                        "chunk_id": f"{table.table_id}:{len(documents)}",
                        "table_id": table.table_id,
                        "table_row": row,
                        "retrieval_score": score,
                    },
                )
            )
        return ComplexExecutionResult(
            documents=documents,
            trace={
                "query_type": "table",
                "table_matches": len(matches),
                "table_ids": sorted({table.table_id for table, _, _ in matches}),
            },
        )
''',
    )

    _write(
        "src/coal_kb/complex_qa/synthesis.py",
        '''"""执行跨文档综合检索并控制单一来源占比。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.utils import clone_plan_for_subquery, deduplicate_documents, tag_documents
from coal_kb.core.models.query import QueryPlan


@dataclass
class CrossDocumentExecutor:
    """持有检索器并聚合不同文档中的支持、冲突和条件证据。"""

    retriever: Any
    min_sources: int
    max_per_source: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        candidates = []
        steps = []
        for subquery in plan.complex.subqueries:
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(clone_plan_for_subquery(plan, subquery), trace=local_trace)
            candidates.extend(tag_documents(hits, complex_route="cross_document", complex_role=subquery.subquery_id))
            steps.append({"subquery_id": subquery.subquery_id, "query": subquery.query, "hits": len(hits)})
        unique = deduplicate_documents(candidates)
        per_source: dict[str, int] = {}
        selected = []
        for document in unique:
            source = str((document.metadata or {}).get("source_file") or "unknown")
            if per_source.get(source, 0) >= self.max_per_source:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            selected.append(document)
        return ComplexExecutionResult(
            documents=selected,
            trace={
                "query_type": "cross_document",
                "steps": steps,
                "source_count": len(per_source),
                "sources": sorted(per_source),
                "minimum_sources_met": len(per_source) >= self.min_sources,
            },
        )
''',
    )

    _write(
        "src/coal_kb/complex_qa/service.py",
        '''"""统一分派事实、比较、多跳、聚合、表格和跨文档路线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.aggregation import AggregationExecutor, AggregationRepository
from coal_kb.complex_qa.comparison import ComparisonExecutor
from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.multi_hop import MultiHopExecutor
from coal_kb.complex_qa.synthesis import CrossDocumentExecutor
from coal_kb.complex_qa.tables import TableExecutor, TableRepository
from coal_kb.core.models.query import QueryPlan


@dataclass
class ComplexQuestionService:
    """持有各路线执行器并提供单一 process() 入口。"""

    retriever: Any
    sqlite_path: str
    table_records_path: str
    comparison_k_per_side: int
    max_multi_hop_steps: int
    aggregation_record_limit: int
    aggregation_evidence_limit: int
    table_top_k: int
    cross_document_min_sources: int
    cross_document_max_per_source: int

    def __post_init__(self) -> None:
        self._comparison = ComparisonExecutor(self.retriever, self.comparison_k_per_side)
        self._multi_hop = MultiHopExecutor(self.retriever, self.max_multi_hop_steps)
        self._aggregation = AggregationExecutor(
            AggregationRepository(self.sqlite_path, self.aggregation_record_limit),
            self.retriever,
            self.aggregation_evidence_limit,
        )
        self._table = TableExecutor(TableRepository(self.table_records_path), self.retriever, self.table_top_k)
        self._cross_document = CrossDocumentExecutor(
            self.retriever,
            self.cross_document_min_sources,
            self.cross_document_max_per_source,
        )

    def process(self, plan: QueryPlan, trace: dict[str, Any] | None = None) -> list[Any]:
        query_type = plan.complex.query_type
        if query_type == "unanswerable":
            result = ComplexExecutionResult(documents=[], trace={"query_type": query_type, "reason": plan.complex.reason})
        elif query_type == "comparison":
            result = self._comparison.process(plan)
        elif query_type == "multi_hop":
            result = self._multi_hop.process(plan)
        elif query_type == "aggregation":
            result = self._aggregation.process(plan)
        elif query_type == "table":
            result = self._table.process(plan)
        elif query_type == "cross_document":
            result = self._cross_document.process(plan)
        else:
            local_trace: dict[str, Any] = {}
            documents = self.retriever.execute(plan, trace=local_trace)
            result = ComplexExecutionResult(documents=documents, trace={"query_type": "fact", "retrieval": local_trace})
        if trace is not None:
            trace["complex_route"] = plan.complex.model_dump()
            trace["complex_execution"] = result.trace
        return result.documents
''',
    )

    _write(
        "src/coal_kb/complex_qa/__init__.py",
        '''"""导出 Milestone C 复杂科学问答的正式接口。"""

from coal_kb.complex_qa.planning import build_complex_spec
from coal_kb.complex_qa.router import route_question
from coal_kb.complex_qa.service import ComplexQuestionService

__all__ = ["ComplexQuestionService", "build_complex_spec", "route_question"]
''',
    )

    _write(
        "src/coal_kb/retrieval/query/planner.py",
        '''"""构建事实检索与复杂科学问答共用的 QueryPlan。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.planning import build_complex_spec
from coal_kb.core.models.query import (
    AnswerSpec,
    Constraint,
    ContextSpec,
    DiversitySpec,
    NeighborSpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RerankSpec,
    RetrievalStep,
)
from coal_kb.retrieval.query.filter_parser import FilterParser


@dataclass
class QueryPlanner:
    """持有领域过滤解析器并生成完整 QueryPlan。"""

    filter_parser: FilterParser

    def build_plan(
        self,
        query: str,
        cfg: Any,
        *,
        enable_llm: bool = False,
        llm_config: Any = None,
    ) -> QueryPlan:
        del enable_llm, llm_config
        normalized = " ".join(query.strip().split())
        parsed = self.filter_parser.parse(normalized)
        hard = [self._constraint(value) for value in parsed.hard_constraints]
        soft = [self._constraint(value) for value in parsed.soft_constraints]
        complex_spec = build_complex_spec(
            normalized,
            max_subqueries=cfg.complex_qa.max_subqueries,
            max_multi_hop_steps=cfg.complex_qa.max_multi_hop_steps,
        )
        retrieval_steps = self._retrieval_steps(cfg)
        context_tokens, evidence_chunks = self._context_budget(complex_spec.query_type, cfg)
        min_evidence = 1 if complex_spec.query_type in {"aggregation", "table"} else 2
        if complex_spec.query_type == "cross_document":
            min_evidence = cfg.complex_qa.cross_document_min_sources
        return QueryPlan(
            query=QueryUnderstanding(raw=query, normalized=normalized, hard_constraints=hard, soft_constraints=soft),
            complex=complex_spec,
            retrieval_steps=retrieval_steps,
            relax_policy=RelaxPolicy(max_steps=cfg.retrieval.max_relax_steps),
            rerank=RerankSpec(enabled=cfg.retrieval.rerank_enabled, top_n=cfg.retrieval.rerank_top_n),
            neighbor=NeighborSpec(enabled=complex_spec.query_type in {"comparison", "multi_hop", "table", "cross_document"}, window=1),
            diversity=DiversitySpec(
                max_per_source=(
                    cfg.complex_qa.cross_document_max_per_source
                    if complex_spec.query_type == "cross_document"
                    else cfg.retrieval.max_per_source
                )
            ),
            context=ContextSpec(max_context_tokens=context_tokens, max_evidence_chunks=evidence_chunks),
            answer=AnswerSpec(min_evidence=min_evidence),
        )

    @staticmethod
    def _constraint(value: Any) -> Constraint:
        return Constraint(
            field=value.name,
            op=value.ctype,
            value=value.value,
            priority=value.priority,
            confidence=value.confidence,
            source=value.source,
        )

    @staticmethod
    def _retrieval_steps(cfg: Any) -> list[RetrievalStep]:
        if cfg.retrieval.two_stage.enabled:
            return [
                RetrievalStep(
                    name="parent_recall",
                    level="parent",
                    fusion_mode="rrf",
                    k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
                    k_final=cfg.retrieval.two_stage.parent_k_final,
                    where_mode="hard_only",
                    enable_relax=False,
                ),
                RetrievalStep(
                    name="child_recall",
                    level="child",
                    fusion_mode="rrf",
                    k_candidates=cfg.retrieval.two_stage.child_k_candidates,
                    k_final=cfg.retrieval.two_stage.child_k_final,
                    where_mode="hard_only",
                    enable_relax=cfg.retrieval.two_stage.allow_relax_in_stage2,
                ),
            ]
        return [
            RetrievalStep(
                name="single_recall",
                level="single",
                fusion_mode="rrf",
                k_candidates=cfg.retrieval.k,
                k_final=cfg.retrieval.k,
                where_mode="hard_only",
                enable_relax=True,
            )
        ]

    @staticmethod
    def _context_budget(query_type: str, cfg: Any) -> tuple[int, int]:
        base_tokens = cfg.complex_qa.base_context_tokens
        base_chunks = cfg.complex_qa.base_evidence_chunks
        multipliers = {
            "comparison": 1.6,
            "multi_hop": 1.8,
            "aggregation": 1.2,
            "table": 1.4,
            "cross_document": 2.0,
        }
        multiplier = multipliers.get(query_type, 1.0)
        return int(base_tokens * multiplier), max(base_chunks, int(base_chunks * multiplier))
''',
    )

    _replace(
        "src/coal_kb/retrieval/query/__init__.py",
        'from .filter_parser import FilterParser\n\n__all__ = ["FilterParser"]\n',
        'from .filter_parser import FilterParser\nfrom .planner import QueryPlanner\n\n__all__ = ["FilterParser", "QueryPlanner"]\n',
    )

    models_path = ROOT / "src/coal_kb/infra/config/models.py"
    models_text = models_path.read_text(encoding="utf-8")
    insertion = '''\n\nclass ComplexQAConfig(BaseModel):\n    """定义 Milestone C 路由、数据源和上下文预算。"""\n\n    enabled: bool = True\n    max_subqueries: int = 4\n    max_multi_hop_steps: int = 3\n    comparison_k_per_side: int = 4\n    cross_document_min_sources: int = 2\n    cross_document_max_per_source: int = 2\n    aggregation_record_limit: int = 500\n    aggregation_evidence_limit: int = 12\n    table_records_path: str = "data/interim/table_records.jsonl"\n    table_top_k: int = 5\n    base_context_tokens: int = 2400\n    base_evidence_chunks: int = 10\n'''
    if "class ComplexQAConfig" not in models_text:
        models_text = models_text.replace("\n\nclass LoggingConfig(BaseModel):", insertion + "\n\nclass LoggingConfig(BaseModel):")
    if "complex_qa: ComplexQAConfig" not in models_text:
        models_text = models_text.replace(
            "    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)\n",
            "    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)\n    complex_qa: ComplexQAConfig = Field(default_factory=ComplexQAConfig)\n",
        )
    models_path.write_text(models_text, encoding="utf-8")

    config_init = ROOT / "src/coal_kb/infra/config/__init__.py"
    init_text = config_init.read_text(encoding="utf-8")
    if "ComplexQAConfig," not in init_text:
        init_text = init_text.replace("    ChunkingProfile,\n", "    ChunkingProfile,\n    ComplexQAConfig,\n")
    if '"ComplexQAConfig",' not in init_text:
        init_text = init_text.replace('    "ChunkingConfig",\n', '    "ChunkingConfig",\n    "ComplexQAConfig",\n')
    config_init.write_text(init_text, encoding="utf-8")


if __name__ == "__main__":
    process()
