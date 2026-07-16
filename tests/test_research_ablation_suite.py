"""验证版本化实验配置、消融矩阵、恢复和指标聚合。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from coal_kb.research.models import ResearchRoute
from coal_kb.research.suites import AblationSuiteRunner, ExperimentSuiteConfig, apply_route_parameters


def _write_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: research-suite.v1",
                "name: test-suite",
                "dataset: dataset.jsonl",
                f"output_dir: {output_dir.as_posix()}",
                "k_values: [1, 3]",
                "repeats: 2",
                "seed: 10",
                "resume: true",
                "variants:",
                "  - name: standard",
                "    route: standard",
                "  - name: graph-small",
                "    route: graph",
                "    parameters:",
                "      graph.max_edges: 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_suite_config_expands_deterministic_matrix(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    _write_config(config_path, tmp_path / "artifacts")

    config = ExperimentSuiteConfig.from_file(config_path)
    first = config.expand_runs()
    second = config.expand_runs()

    assert config.version == "research-suite.v1"
    assert len(first) == 4
    assert [run.run_id for run in first] == [run.run_id for run in second]
    assert first[0].seed == 10
    assert first[1].seed == 11
    assert first[2].variant.route is ResearchRoute.GRAPH
    assert first[2].variant.parameters == {"graph.max_edges": 12}


def test_ablation_runner_writes_manifest_aggregates_and_resumes(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    output_dir = tmp_path / "artifacts"
    _write_config(config_path, output_dir)
    config = ExperimentSuiteConfig.from_file(config_path)
    calls: list[str] = []

    def run(spec):
        calls.append(spec.name)
        metric = 1.0 if spec.route is ResearchRoute.STANDARD else 2.0
        manifest = {"experiment_id": spec.name, "metrics": {"retrieval": {"recall_at_1": metric}}}
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "experiment.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    first = AblationSuiteRunner(config=config, run_fn=run).process()
    second = AblationSuiteRunner(config=config, run_fn=run).process()

    assert len(calls) == 4
    assert first["completed_count"] == 4
    assert second["completed_count"] == 4
    assert {record["status"] for record in second["runs"]} == {"resumed"}
    assert first["aggregates"]["standard"]["metrics"]["retrieval.recall_at_1"] == {
        "mean": 1.0,
        "std": 0.0,
        "values": [1.0, 1.0],
    }
    assert (output_dir / "suite_manifest.json").is_file()
    assert (output_dir / "suite_results.json").is_file()
    assert (output_dir / "summary.md").is_file()


def test_route_parameters_are_whitelisted() -> None:
    service = SimpleNamespace(
        graph_route=SimpleNamespace(seed_count=3, max_edges=80),
        multimodal_route=SimpleNamespace(requested_boost=1.5, secondary_boost=0.15),
        agent_route=SimpleNamespace(max_steps=3),
    )

    apply_route_parameters(
        service,
        {
            "graph.max_edges": 20,
            "multimodal.secondary_boost": 0.0,
            "agent.max_steps": 2,
        },
    )

    assert service.graph_route.max_edges == 20
    assert service.multimodal_route.secondary_boost == 0.0
    assert service.agent_route.max_steps == 2
