"""将标注 JSONL 确定性拆分为 LoRA 训练集与验证集。"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from coal_kb.data_preparation import config


@dataclass
class BuildLoraDataset:
    input_path: Path
    output_dir: Path
    val_ratio: float

    @staticmethod
    def _load_rows(path: Path, *, desc: str) -> list[dict[str, Any]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        for line in tqdm(lines, total=len(lines), desc=desc):
            if not line.strip():
                continue
            item: dict[str, Any] = json.loads(line)
            if all(key in item for key in ("instruction", "input", "output")):
                rows.append(item)
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in rows)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def process(self) -> dict[str, int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._load_rows(self.input_path, desc=self.__class__.__name__)

        # 使用独立随机实例，避免污染全局随机状态。
        rng = random.Random(config.SAMPLE_SEED)
        rng.shuffle(rows)

        validation_count = max(1, int(len(rows) * self.val_ratio)) if rows else 0
        validation_rows = rows[:validation_count]
        training_rows = rows[validation_count:]

        self._write_jsonl(self.output_dir / "lora_train.jsonl", training_rows)
        self._write_jsonl(self.output_dir / "lora_val.jsonl", validation_rows)
        return {"train": len(training_rows), "val": len(validation_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic LoRA train/validation datasets.")
    parser.add_argument("--in", dest="input_path", default="data/artifacts/curated_pairs.jsonl")
    parser.add_argument("--outdir", default="data/artifacts")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=config.SAMPLE_SEED, help="兼容参数，只允许固定种子。")
    args = parser.parse_args()
    if args.seed != config.SAMPLE_SEED:
        parser.error(f"随机采样种子固定为 {config.SAMPLE_SEED}，不允许覆盖。")

    step = BuildLoraDataset(
        input_path=Path(args.input_path),
        output_dir=Path(args.outdir),
        val_ratio=float(args.val_ratio),
    )
    print(step.process())


if __name__ == "__main__":
    main()

# 运行命令：python scripts/build_lora_dataset.py
