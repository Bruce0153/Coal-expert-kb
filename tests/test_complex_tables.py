"""验证标准表格 JSONL 的行检索和精确证据元数据。"""

from __future__ import annotations

import json

from coal_kb.complex_qa.tables import TableExecutor, TableRepository
from coal_kb.core.models.query import ComplexQuestionSpec, QueryPlan, QueryUnderstanding


class EmptyRetriever:
    def execute(self, plan, trace=None):
        return []


def test_table_executor_returns_matching_row(tmp_path) -> None:
    path = tmp_path / "tables.jsonl"
    payload = {
        "table_id": "table_001",
        "source_file": "paper.pdf",
        "page": 7,
        "caption": "不同温度下的H2产率",
        "headers": ["T_K", "H2"],
        "rows": [{"T_K": 1000, "H2": 30.0}, {"T_K": 1200, "H2": 42.1}],
        "nearby_text": "蒸汽气化实验",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    plan = QueryPlan(
        query=QueryUnderstanding(raw="表格中1200K的H2产率", normalized="表格中1200K的H2产率"),
        complex=ComplexQuestionSpec(query_type="table", require_table=True),
    )
    result = TableExecutor(TableRepository(str(path)), EmptyRetriever(), 5).process(plan)
    assert result.documents
    assert result.documents[0].metadata["table_id"] == "table_001"
    assert result.documents[0].metadata["page"] == 7
    assert result.trace["table_matches"] >= 1
