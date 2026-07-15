"""修正复杂评估入口、模板转义和静态类型边界。"""

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
    tables_text = tables_text.replace(
        "        documents = []\n",
        "        documents: list[Document] = []\n",
    )
    tables_path.write_text(tables_text, encoding="utf-8")

    table_test_path = Path("tests/test_complex_tables.py")
    table_test_text = table_test_path.read_text(encoding="utf-8")
    table_test_text = table_test_text.replace(
        'json.dumps(payload, ensure_ascii=False) + "\n",',
        'json.dumps(payload, ensure_ascii=False) + "\\n",',
    )
    table_test_path.write_text(table_test_text, encoding="utf-8")

    planning_path = Path("src/coal_kb/complex_qa/planning.py")
    planning_text = planning_path.read_text(encoding="utf-8")
    planning_text = planning_text.replace(
        "        templates = (\n",
        "        multi_hop_templates: tuple[tuple[str, str, str, list[str]], ...] = (\n",
        1,
    )
    planning_text = planning_text.replace(
        "        for subquery_id, subquery, purpose, dependencies in templates[:max_multi_hop_steps]:\n",
        "        for subquery_id, subquery, purpose, dependencies in multi_hop_templates[:max_multi_hop_steps]:\n",
    )
    planning_text = planning_text.replace(
        "        templates = (\n",
        "        cross_document_templates: tuple[tuple[str, str, str], ...] = (\n",
        1,
    )
    planning_text = planning_text.replace(
        "        subqueries = [SubQuerySpec(subquery_id=item[0], query=item[1], purpose=item[2]) for item in templates]\n",
        "        subqueries = [\n            SubQuerySpec(subquery_id=item[0], query=item[1], purpose=item[2])\n            for item in cross_document_templates\n        ]\n",
    )
    planning_path.write_text(planning_text, encoding="utf-8")

    aggregation_path = Path("src/coal_kb/complex_qa/aggregation.py")
    aggregation_text = aggregation_path.read_text(encoding="utf-8")
    aggregation_text = aggregation_text.replace(
        "                value = self._value(record, key)\n                label = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value or \"unknown\")\n",
        "                group_value = self._value(record, key)\n                label = (\n                    json.dumps(group_value, ensure_ascii=False, sort_keys=True)\n                    if isinstance(group_value, (dict, list))\n                    else str(group_value or \"unknown\")\n                )\n",
    )
    old_numeric = '''        numeric = [(record, self._numeric_value(record, field)) for record in records]
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
'''
    new_numeric = '''        raw_numeric = [(record, self._numeric_value(record, field)) for record in records]
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
'''
    if old_numeric not in aggregation_text:
        raise ValueError("未找到聚合数值类型收窄代码块")
    aggregation_path.write_text(aggregation_text.replace(old_numeric, new_numeric), encoding="utf-8")


if __name__ == "__main__":
    process()
