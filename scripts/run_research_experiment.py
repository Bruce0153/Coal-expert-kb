"""运行 Milestone D 标准或 Graph 路线研究实验。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coal_kb.application.ask import AskRuntime, build_runtime
from coal_kb.evaluation.models import EvaluationCase
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.research import ExperimentSpec, ResearchExperiment, ResearchRoute


@dataclass
class RunResearchExperiment:
    """组装运行时并执行固定路线实验。"""

    cfg: AppConfig
    spec: ExperimentSpec
    runtime: AskRuntime = field(init=False)

    def _retrieve(self, case: EvaluationCase, k: int):
        plan = self.runtime.planner.build_plan(
            case.query,
            self.cfg,
            enable_llm=False,
            llm_config=None,
        )
        trace: dict[str, Any] = {}
        documents = self.runtime.research_route_service.process(
            plan,
            route=self.spec.route,
            trace=trace,
        )
        return documents[:k], trace

    def process(self) -> dict[str, Any]:
        self.runtime = build_runtime(
            self.cfg,
            backend=self.cfg.backend,
            k=max(self.spec.k_values),
            enable_llm=False,
        )
        return ResearchExperiment(
            spec=self.spec,
            retrieve_fn=self._retrieve,
        ).process()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible Milestone D experiment.")
    parser.add_argument("--name", default="milestone-d-baseline")
    parser.add_argument("--route", choices=[route.value for route in ResearchRoute], default="standard")
    parser.add_argument("--dataset", default="data/eval/evaluation_sample.jsonl")
    parser.add_argument("--output-dir", default="data/artifacts/research_experiment")
    parser.add_argument("--k", default="1,3,5,10")
    args = parser.parse_args()
    k_values = tuple(sorted({int(value) for value in args.k.split(",") if value.strip()}))
    if not k_values or min(k_values) < 1:
        raise ValueError("At least one positive k value is required")
    RunResearchExperiment(
        cfg=load_config(),
        spec=ExperimentSpec(
            name=args.name,
            route=ResearchRoute(args.route),
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            k_values=k_values,
        ),
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/run_research_experiment.py --route graph
