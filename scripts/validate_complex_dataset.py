"""验证复杂科学问答 JSONL 的格式、唯一性和类型覆盖。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from coal_kb.evaluation.models import EvaluationCase, QueryType
from coal_kb.utils.jsonl import iter_jsonl


@dataclass
class ValidateComplexDataset:
    """持有数据集路径并执行完整格式验证。"""

    dataset_path: Path
    require_all_types: bool = False

    def process(self) -> dict[str, object]:
        rows = list(iter_jsonl(self.dataset_path))
        cases = [
            EvaluationCase.from_dict(payload, row_number=row_number)
            for row_number, payload in tqdm(
                rows,
                total=len(rows),
                desc=self.__class__.__name__,
            )
        ]
        identifiers = [case.case_id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("复杂问答评估集中的 id 必须唯一")

        counts: dict[str, int] = {}
        for case in cases:
            counts[case.query_type.value] = counts.get(case.query_type.value, 0) + 1

        if self.require_all_types:
            required = {
                QueryType.COMPARISON.value,
                QueryType.MULTI_HOP.value,
                QueryType.AGGREGATION.value,
                QueryType.TABLE.value,
                QueryType.CROSS_DOCUMENT.value,
            }
            missing = sorted(required - set(counts))
            if missing:
                raise ValueError(f"复杂问答评估集缺少类型: {missing}")

        summary = {
            "case_count": len(cases),
            "query_type_counts": counts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a complex science QA JSONL dataset."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/complex_science_sample.jsonl",
    )
    parser.add_argument("--require-all-types", action="store_true")
    args = parser.parse_args()
    ValidateComplexDataset(
        Path(args.dataset),
        require_all_types=args.require_all_types,
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/validate_complex_dataset.py --require-all-types
