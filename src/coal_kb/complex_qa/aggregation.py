"""在结构化实验记录上执行可复现统计并生成证据文档。"""

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
                group_value = self._value(record, key)
                label = (
                    json.dumps(group_value, ensure_ascii=False, sort_keys=True)
                    if isinstance(group_value, (dict, list))
                    else str(group_value or "unknown")
                )
                groups[label] = groups.get(label, 0) + 1
            return AggregationResult(operation=operation, field=key, value=groups, sample_size=len(records), records=tuple(records))

        raw_numeric = [(record, self._numeric_value(record, field)) for record in records]
        numeric: list[tuple[dict[str, Any], float]] = [
            (record, numeric_value)
            for record, numeric_value in raw_numeric
            if numeric_value is not None
        ]
        if not numeric:
            return AggregationResult(operation=operation, field=field, value=None, sample_size=0, records=())
        values: list[float] = [numeric_value for _, numeric_value in numeric]
        selected_records = [record for record, _ in numeric]
        aggregate_value: Any
        if operation == "sum":
            aggregate_value = sum(values)
        elif operation == "average":
            aggregate_value = statistics.fmean(values)
        elif operation == "median":
            aggregate_value = statistics.median(values)
        elif operation == "min":
            aggregate_value = min(values)
        elif operation == "max":
            aggregate_value = max(values)
        elif operation == "top_k":
            ranked = sorted(numeric, key=lambda item: item[1], reverse=True)[:top_k]
            aggregate_value = [
                {"record_id": record.get("record_id"), "value": score}
                for record, score in ranked
            ]
            selected_records = [record for record, _ in ranked]
        else:
            aggregate_value = len(values)
        return AggregationResult(
            operation=operation,
            field=field,
            value=aggregate_value,
            sample_size=len(values),
            records=tuple(selected_records),
        )

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
