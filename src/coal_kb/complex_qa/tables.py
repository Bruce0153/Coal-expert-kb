"""读取标准表格记录并执行表格、行和单元格检索。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.complex_qa.models import ComplexExecutionResult, TableRecord
from coal_kb.core.models.query import QueryPlan
from coal_kb.utils.jsonl import read_jsonl


@dataclass
class TableRepository:
    """从 JSONL 表格资产中加载并检索记录。"""

    records_path: str

    def load(self) -> list[TableRecord]:
        path = Path(self.records_path)
        if not path.exists():
            return []
        return [
            TableRecord.model_validate(payload)
            for payload in read_jsonl(path)
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[tuple[TableRecord, dict[str, Any], float]]:
        query_terms = self._terms(query)
        candidates: list[tuple[TableRecord, dict[str, Any], float]] = []
        for table in self.load():
            for row in table.rows:
                searchable = " ".join(
                    [
                        table.caption,
                        " ".join(table.headers),
                        table.nearby_text,
                        json.dumps(row, ensure_ascii=False),
                    ]
                )
                score = len(query_terms & self._terms(searchable)) / max(
                    1,
                    len(query_terms),
                )
                if score > 0:
                    candidates.append((table, row, score))
        return sorted(
            candidates,
            key=lambda item: (-item[2], item[0].table_id),
        )[:top_k]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_.%+-]+|[一-鿿]", text.lower()))


@dataclass
class TableExecutor:
    """执行表格路线并在没有表格资产时回退到文档检索。"""

    repository: TableRepository
    retriever: Any
    top_k: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        matches = self.repository.search(
            plan.query.normalized,
            top_k=self.top_k,
        )
        if not matches:
            fallback = self.retriever.execute(plan, trace={})
            table_documents = [
                document
                for document in fallback
                if "table"
                in str((document.metadata or {}).get("section", "")).lower()
            ]
            return ComplexExecutionResult(
                documents=table_documents or fallback,
                trace={
                    "query_type": "table",
                    "fallback": "document_retrieval",
                    "table_matches": 0,
                },
            )

        documents = [
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
                    "chunk_id": f"{table.table_id}:{index}",
                    "table_id": table.table_id,
                    "table_row": row,
                    "retrieval_score": score,
                },
            )
            for index, (table, row, score) in enumerate(matches)
        ]
        return ComplexExecutionResult(
            documents=documents,
            trace={
                "query_type": "table",
                "table_matches": len(matches),
                "table_ids": sorted(
                    {table.table_id for table, _, _ in matches}
                ),
            },
        )
