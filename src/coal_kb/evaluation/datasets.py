"""读取、验证并写出评估数据集。"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from coal_kb.evaluation.models import EvaluationCase
from coal_kb.utils.jsonl import iter_jsonl, write_jsonl


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """从 JSONL 加载并验证评估案例。"""
    rows = list(iter_jsonl(path))
    cases: list[EvaluationCase] = []
    for row_number, payload in tqdm(
        rows,
        total=len(rows),
        desc="EvaluationDataset",
    ):
        if not isinstance(payload, dict):
            raise TypeError(f"Evaluation row {row_number} must be a JSON object")
        cases.append(EvaluationCase.from_dict(payload, row_number=row_number))

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case ids must be unique")
    return cases


def save_evaluation_cases(
    path: str | Path,
    cases: list[EvaluationCase],
) -> None:
    """以稳定 JSONL 格式保存评估案例。"""
    write_jsonl(path, (case.to_dict() for case in cases))
