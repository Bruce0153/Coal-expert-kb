"""写出机器可读和人工可读的评估产物。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coal_kb.evaluation import config
from coal_kb.evaluation.models import CaseEvaluationResult


@dataclass
class EvaluationReportWriter:
    """保存评估产物到单一输出目录。"""

    output_dir: Path

    def process(self, *, metrics: dict[str, Any], results: list[CaseEvaluationResult], manifest: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        failures = [result for result in results if result.failure_category != config.FAILURE_NONE]
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_jsonl(self.output_dir / "case_results.jsonl", [result.to_dict() for result in results])
        self._write_jsonl(self.output_dir / "failures.jsonl", [result.to_dict() for result in failures])
        manifest_payload = {
            "manifest_version": config.MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **manifest,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "summary.md").write_text(self._summary(metrics, results, failures), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @staticmethod
    def _summary(metrics: dict[str, Any], results: list[CaseEvaluationResult], failures: list[CaseEvaluationResult]) -> str:
        lines = ["# Evaluation Summary", "", f"- Cases: {len(results)}", f"- Failures: {len(failures)}", ""]
        for group_name in ("retrieval", "answer"):
            values = metrics.get(group_name) or {}
            lines.append(f"## {group_name.title()}")
            lines.append("")
            for key, value in sorted(values.items()):
                lines.append(f"- `{key}`: {value:.4f}")
            lines.append("")
        lines.extend(["## Failure Categories", ""])
        categories: dict[str, int] = {}
        for result in failures:
            categories[result.failure_category] = categories.get(result.failure_category, 0) + 1
        for category, count in sorted(categories.items()):
            lines.append(f"- `{category}`: {count}")
        return "\n".join(lines) + "\n"
