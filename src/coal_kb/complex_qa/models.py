"""定义复杂科学问答执行结果、聚合结果和标准表格记录。"""

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
