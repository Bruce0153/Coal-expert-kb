"""修正复杂评估脚本入口和问题类型指标引用。"""

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


if __name__ == "__main__":
    process()
