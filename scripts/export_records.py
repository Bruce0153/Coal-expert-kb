from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from coal_kb.logging import setup_logging
from coal_kb.settings import load_config
from coal_kb.store.sql_store import SQLiteStore

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ExperimentRecords from SQLite to CSV.")
    parser.add_argument("--out", default="data/artifacts/records.csv")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SQLiteStore(cfg.paths.sqlite_path)
    rows = db.list_records(limit=args.limit)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "record_id": r.record_id,
                    "source_file": r.source_file,
                    "stage": r.stage,
                    "coal_name": r.coal_name,
                    "reactor_type": r.reactor_type,
                    "T_K": r.T_K,
                    "P_MPa": r.P_MPa,
                    "gas_agent_json": r.gas_agent_json,
                    "ratios_json": r.ratios_json,
                    "pollutants_json": r.pollutants_json,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                }
            )

    print(f"✅ exported: {out_path}")


if __name__ == "__main__":
    main()
