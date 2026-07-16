"""复用统一评估管线执行并记录研究实验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from coal_kb.evaluation.pipeline import AnswerFunction, EvaluationPipeline, RetrieveFunction
from coal_kb.research.models import ExperimentSpec
from coal_kb.utils.hash import stable_chunk_id


@dataclass
class ResearchExperiment:
    """固定路线、数据集和参数并输出可复现实验清单。"""

    spec: ExperimentSpec
    retrieve_fn: RetrieveFunction
    answer_fn: AnswerFunction | None = None

    def process(self) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        experiment_id = stable_chunk_id(
            self.spec.name,
            self.spec.route.value,
            started_at.isoformat(),
        )
        metrics = EvaluationPipeline(
            dataset_path=self.spec.dataset_path,
            output_dir=self.spec.output_dir,
            retrieve_fn=self.retrieve_fn,
            answer_fn=self.answer_fn,
            k_values=self.spec.k_values,
            run_metadata={
                "experiment_id": experiment_id,
                "experiment_name": self.spec.name,
                "research_route": self.spec.route.value,
                "started_at": started_at.isoformat(),
                **self.spec.metadata,
            },
        ).process()
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "experiment_id": experiment_id,
            "name": self.spec.name,
            "route": self.spec.route.value,
            "dataset_path": str(self.spec.dataset_path),
            "output_dir": str(self.spec.output_dir),
            "k_values": list(self.spec.k_values),
            "metadata": self.spec.metadata,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 6),
            "metrics": metrics,
        }
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.spec.output_dir / "experiment.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return manifest
