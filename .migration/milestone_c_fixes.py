"""修正复杂评估入口、问题类型引用和模板换行转义。"""

from pathlib import Path


def process() -> None:
    script_path = Path("scripts/validate_complex_dataset.py")
    script_text = script_path.read_text(encoding="utf-8")
    script_text = script_text.replace(
        "# 运行命令：python scripts/validate_complex_dataset.py --require-all-types",
        "# 运行命令：PYTHONPATH=src python scripts/validate_complex_dataset.py --require-all-types",
    )
    script_path.write_text(script_text, encoding="utf-8")

    metrics_path = Path("src/coal_kb/evaluation/metrics.py")
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_text = metrics_text.replace(
        "from coal_kb.evaluation.models import AnswerObservation, EvidenceReference, EvaluationCase, RetrievedEvidence",
        "from coal_kb.evaluation.models import AnswerObservation, EvidenceReference, EvaluationCase, QueryType, RetrievedEvidence",
    )
    metrics_text = metrics_text.replace(
        'expected_type = "fact" if case.query_type == case.query_type.CONDITION else case.query_type.value',
        'expected_type = "fact" if case.query_type == QueryType.CONDITION else case.query_type.value',
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")

    tables_path = Path("src/coal_kb/complex_qa/tables.py")
    tables_text = tables_path.read_text(encoding="utf-8")
    tables_text = tables_text.replace(
        'f"表格标题：{table.caption}\n"',
        'f"表格标题：{table.caption}\\n"',
    )
    tables_text = tables_text.replace(
        'f"表头：{json.dumps(table.headers, ensure_ascii=False)}\n"',
        'f"表头：{json.dumps(table.headers, ensure_ascii=False)}\\n"',
    )
    tables_path.write_text(tables_text, encoding="utf-8")

    table_test_path = Path("tests/test_complex_tables.py")
    table_test_text = table_test_path.read_text(encoding="utf-8")
    table_test_text = table_test_text.replace(
        'json.dumps(payload, ensure_ascii=False) + "\n",',
        'json.dumps(payload, ensure_ascii=False) + "\\n",',
    )
    table_test_path.write_text(table_test_text, encoding="utf-8")


if __name__ == "__main__":
    process()
