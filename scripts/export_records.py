"""从 SQLite 导出实验记录为 CSV 文件。"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.infra.persistence.sql import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class ExportRecords:
    cfg: AppConfig
    output_path: Path
    limit: int

    @staticmethod
    def _serialize_record(record: Any) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "source_file": record.source_file,
            "stage": record.stage,
            "coal_name": record.coal_name,
            "reactor_type": record.reactor_type,
            "T_K": record.T_K,
            "P_MPa": record.P_MPa,
            "gas_agent_json": record.gas_agent_json,
            "ratios_json": record.ratios_json,
            "pollutants_json": record.pollutants_json,
            "created_at": record.created_at.isoformat() if record.created_at else "",
            "updated_at": record.updated_at.isoformat() if record.updated_at else "",
        }

    def process(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        database = SQLiteStore(self.cfg.paths.sqlite_path)
        records = database.list_records(limit=self.limit)
        fieldnames = list(self._serialize_record(records[0]).keys()) if records else [
            "record_id",
            "source_file",
            "stage",
            "coal_name",
            "reactor_type",
            "T_K",
            "P_MPa",
            "gas_agent_json",
            "ratios_json",
            "pollutants_json",
            "created_at",
            "updated_at",
        ]

        with self.output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in tqdm(records, total=len(records), desc=self.__class__.__name__):
                writer.writerow(self._serialize_record(record))
        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ExperimentRecords from SQLite to CSV.")
    parser.add_argument("--out", default="data/artifacts/records.csv")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    step = ExportRecords(cfg=cfg, output_path=Path(args.out), limit=args.limit)
    print(f"exported: {step.process()}")


if __name__ == "__main__":
    main()

# 运行命令：python scripts/export_records.py
