"""从版本化配置运行可恢复的 Milestone D 消融实验套件。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coal_kb.application.ask import AskRuntime, build_runtime
from coal_kb.evaluation.models import EvaluationCase
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.research import ExperimentSpec, ResearchExperiment
from coal_kb.research.suites import AblationSuiteRunner, ExperimentSuiteConfig, apply_route_parameters


@dataclass
class RunAblationSuite:
    """为每个消融运行构建隔离运行时并复用统一评估管线。"""

    cfg: AppConfig
    suite: ExperimentSuiteConfig
    _runtime: AskRuntime | None = field(default=None, init=False)
    _spec: ExperimentSpec | None = field(default=None, init=False)

    def _retrieve(self, case: EvaluationCase, k: int):
        if self._runtime is None or self._spec is None:
            raise RuntimeError("Ablation runtime has not been initialized")
        plan = self._runtime.planner.build_plan(
            case.query,
            self.cfg,
            enable_llm=False,
            llm_config=None,
        )
        trace: dict[str, Any] = {}
        documents = self._runtime.research_route_service.process(
            plan,
            route=self._spec.route,
            trace=trace,
        )
        return documents[:k], trace

    def _run(self, spec: ExperimentSpec) -> dict[str, Any]:
        self._spec = spec
        self._runtime = build_runtime(
            self.cfg,
            backend=self.cfg.backend,
            k=max(spec.k_values),
            enable_llm=False,
        )
        parameters = spec.metadata.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Experiment parameters must be a mapping")
        apply_route_parameters(self._runtime.research_route_service, parameters)
        return ResearchExperiment(spec=spec, retrieve_fn=self._retrieve).process()

    def process(self) -> dict[str, Any]:
        return AblationSuiteRunner(config=self.suite, run_fn=self._run).process()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible and resumable research ablation suite.")
    parser.add_argument("--config", default="configs/research/milestone_d_ablation.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    RunAblationSuite(
        cfg=load_config(),
        suite=ExperimentSuiteConfig.from_file(config_path),
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/run_ablation_suite.py --config configs/research/milestone_d_ablation.yaml
