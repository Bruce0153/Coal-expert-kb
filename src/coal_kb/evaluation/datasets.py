"""读取、验证并写出版本化评估数据集。"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from coal_kb.evaluation.models import EvaluationCase


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """从 JSONL 加载并验证评估案例。"""
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    cases: list[EvaluationCase] = []
    for row_number, line in enumerate(tqdm(lines, total=len(lines), desc="EvaluationDataset"), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"Evaluation row {row_number} must be a JSON object")
        cases.append(EvaluationCase.from_dict(payload, row_number=row_number))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case ids must be unique")
    return cases


def save_evaluation_cases(path: str | Path, cases: list[EvaluationCase]) -> None:
    """以稳定 JSONL 格式保存评估案例。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) for case in cases]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
