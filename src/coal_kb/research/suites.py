"""加载、展开并执行可恢复的研究实验与消融套件。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Mapping

import yaml

from coal_kb.research.models import ExperimentSpec, ResearchRoute

SUITE_CONFIG_VERSION = "research-suite.v1"
SuiteRunFunction = Callable[[ExperimentSpec], dict[str, Any]]


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True)
class ExperimentVariant:
    """定义一个路线及其可审计消融参数。"""

    name: str
    route: ResearchRoute
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteRun:
    """表示套件展开后的单次确定性运行。"""

    run_id: str
    variant: ExperimentVariant
    repeat: int
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    """保存版本化实验配置并展开消融运行矩阵。"""

    name: str
    dataset_path: Path
    output_dir: Path
    variants: tuple[ExperimentVariant, ...]
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    repeats: int = 1
    seed: int = 2026
    resume: bool = True
    fail_fast: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = SUITE_CONFIG_VERSION

    @classmethod
    def from_file(cls, path: Path) -> ExperimentSuiteConfig:
        """从 YAML 或 JSON 加载配置，并相对配置文件解析路径。"""

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Research suite config must be a mapping")
        version = str(payload.get("version") or "")
        if version != SUITE_CONFIG_VERSION:
            raise ValueError(f"Unsupported research suite config version: {version or '<missing>'}")
        base_dir = path.parent.resolve()
        raw_variants = payload.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError("Research suite requires at least one variant")
        variants: list[ExperimentVariant] = []
        names: set[str] = set()
        for raw in raw_variants:
            if not isinstance(raw, dict):
                raise ValueError("Each research variant must be a mapping")
            name = str(raw.get("name") or "").strip()
            if not name:
                raise ValueError("Research variant name cannot be empty")
            if name in names:
                raise ValueError(f"Duplicate research variant name: {name}")
            names.add(name)
            parameters = raw.get("parameters") or {}
            metadata = raw.get("metadata") or {}
            if not isinstance(parameters, dict) or not isinstance(metadata, dict):
                raise ValueError(f"Variant {name} parameters and metadata must be mappings")
            variants.append(
                ExperimentVariant(
                    name=name,
                    route=ResearchRoute(str(raw.get("route") or "standard")),
                    parameters=dict(parameters),
                    metadata=dict(metadata),
                )
            )
        k_values = tuple(sorted({int(value) for value in payload.get("k_values", (1, 3, 5, 10))}))
        repeats = int(payload.get("repeats", 1))
        if not k_values or min(k_values) < 1:
            raise ValueError("Research suite k_values must contain positive integers")
        if repeats < 1:
            raise ValueError("Research suite repeats must be positive")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Research suite metadata must be a mapping")
        return cls(
            name=str(payload.get("name") or path.stem),
            dataset_path=_resolve_path(str(payload.get("dataset") or ""), base_dir),
            output_dir=_resolve_path(str(payload.get("output_dir") or "artifacts"), base_dir),
            variants=tuple(variants),
            k_values=k_values,
            repeats=repeats,
            seed=int(payload.get("seed", 2026)),
            resume=bool(payload.get("resume", True)),
            fail_fast=bool(payload.get("fail_fast", False)),
            metadata=dict(metadata),
            version=version,
        )

    def canonical_payload(self) -> dict[str, Any]:
        """返回不依赖运行时间的规范配置快照。"""

        return {
            "version": self.version,
            "name": self.name,
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "k_values": list(self.k_values),
            "repeats": self.repeats,
            "seed": self.seed,
            "resume": self.resume,
            "fail_fast": self.fail_fast,
            "metadata": self.metadata,
            "variants": [
                {
                    "name": variant.name,
                    "route": variant.route.value,
                    "parameters": variant.parameters,
                    "metadata": variant.metadata,
                }
                for variant in self.variants
            ],
        }

    @property
    def config_digest(self) -> str:
        """计算可用于恢复和对比的配置摘要。"""

        return _json_digest(self.canonical_payload())

    def expand_runs(self) -> tuple[SuiteRun, ...]:
        """按 variant × repeat 展开稳定、可重放的运行列表。"""

        runs: list[SuiteRun] = []
        for variant in self.variants:
            for repeat in range(self.repeats):
                seed = self.seed + repeat
                identity = {
                    "config_digest": self.config_digest,
                    "variant": variant.name,
                    "route": variant.route.value,
                    "parameters": variant.parameters,
                    "repeat": repeat,
                    "seed": seed,
                }
                run_id = _json_digest(identity)[:16]
                runs.append(
                    SuiteRun(
                        run_id=run_id,
                        variant=variant,
                        repeat=repeat,
                        seed=seed,
                        output_dir=self.output_dir / f"{variant.name}__r{repeat + 1}_{run_id}",
                    )
                )
        return tuple(runs)


@dataclass
class AblationSuiteRunner:
    """执行实验矩阵，持久化状态并聚合数值指标。"""

    config: ExperimentSuiteConfig
    run_fn: SuiteRunFunction

    def process(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        suite_manifest = self._suite_manifest()
        self._write_json(self.config.output_dir / "suite_manifest.json", suite_manifest)
        run_records: list[dict[str, Any]] = []
        for run in self.config.expand_runs():
            try:
                record = self._run_or_resume(run)
            except Exception as exc:
                record = self._failure_record(run, exc)
                self._write_json(run.output_dir / "run_status.json", record)
                if self.config.fail_fast:
                    run_records.append(record)
                    break
            run_records.append(record)
        summary = {
            "suite": suite_manifest,
            "run_count": len(run_records),
            "completed_count": sum(record["status"] in {"completed", "resumed"} for record in run_records),
            "failed_count": sum(record["status"] == "failed" for record in run_records),
            "runs": run_records,
            "aggregates": self._aggregate(run_records),
        }
        self._write_json(self.config.output_dir / "suite_results.json", summary)
        (self.config.output_dir / "summary.md").write_text(self._summary_markdown(summary), encoding="utf-8")
        return summary

    def _run_or_resume(self, run: SuiteRun) -> dict[str, Any]:
        experiment_path = run.output_dir / "experiment.json"
        status_path = run.output_dir / "run_status.json"
        if self.config.resume and experiment_path.is_file():
            manifest = json.loads(experiment_path.read_text(encoding="utf-8"))
            record = self._success_record(run, manifest, status="resumed")
            self._write_json(status_path, record)
            return record
        run.output_dir.mkdir(parents=True, exist_ok=True)
        spec = ExperimentSpec(
            name=f"{self.config.name}:{run.variant.name}:r{run.repeat + 1}",
            route=run.variant.route,
            dataset_path=self.config.dataset_path,
            output_dir=run.output_dir,
            k_values=self.config.k_values,
            metadata={
                **self.config.metadata,
                **run.variant.metadata,
                "suite_name": self.config.name,
                "suite_config_version": self.config.version,
                "suite_config_digest": self.config.config_digest,
                "suite_run_id": run.run_id,
                "variant": run.variant.name,
                "repeat": run.repeat,
                "seed": run.seed,
                "parameters": run.variant.parameters,
            },
        )
        manifest = self.run_fn(spec)
        record = self._success_record(run, manifest, status="completed")
        self._write_json(status_path, record)
        return record

    def _suite_manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": self.config.version,
            "config_digest": self.config.config_digest,
            "name": self.config.name,
            "dataset_path": str(self.config.dataset_path),
            "output_dir": str(self.config.output_dir),
            "git_sha": os.getenv("COAL_KB_GIT_SHA") or os.getenv("GITHUB_SHA") or "unknown",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "run_ids": [run.run_id for run in self.config.expand_runs()],
            "config": self.config.canonical_payload(),
        }

    @staticmethod
    def _success_record(run: SuiteRun, manifest: dict[str, Any], *, status: str) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "variant": run.variant.name,
            "route": run.variant.route.value,
            "repeat": run.repeat,
            "seed": run.seed,
            "parameters": run.variant.parameters,
            "output_dir": str(run.output_dir),
            "status": status,
            "metrics": manifest.get("metrics") or {},
            "experiment_id": manifest.get("experiment_id"),
        }

    @staticmethod
    def _failure_record(run: SuiteRun, exc: Exception) -> dict[str, Any]:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": run.run_id,
            "variant": run.variant.name,
            "route": run.variant.route.value,
            "repeat": run.repeat,
            "seed": run.seed,
            "parameters": run.variant.parameters,
            "output_dir": str(run.output_dir),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "metrics": {},
        }

    @classmethod
    def _aggregate(cls, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record["status"] not in {"completed", "resumed"}:
                continue
            by_variant.setdefault(str(record["variant"]), []).append(record)
        aggregates: dict[str, dict[str, Any]] = {}
        for variant, items in sorted(by_variant.items()):
            values: dict[str, list[float]] = {}
            for item in items:
                for key, value in cls._flatten_numeric(item.get("metrics") or {}).items():
                    values.setdefault(key, []).append(value)
            aggregates[variant] = {
                "run_count": len(items),
                "metrics": {
                    key: {
                        "mean": round(fmean(numbers), 8),
                        "std": round(pstdev(numbers), 8) if len(numbers) > 1 else 0.0,
                        "values": numbers,
                    }
                    for key, numbers in sorted(values.items())
                },
            }
        return aggregates

    @classmethod
    def _flatten_numeric(cls, payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                output[path] = float(value)
            elif isinstance(value, Mapping):
                output.update(cls._flatten_numeric(value, path))
        return output

    @staticmethod
    def _summary_markdown(summary: dict[str, Any]) -> str:
        lines = [
            f"# Research Suite: {summary['suite']['name']}",
            "",
            f"- Config digest: `{summary['suite']['config_digest']}`",
            f"- Runs: {summary['run_count']}",
            f"- Completed: {summary['completed_count']}",
            f"- Failed: {summary['failed_count']}",
            "",
            "## Variants",
            "",
        ]
        for variant, aggregate in summary["aggregates"].items():
            lines.append(f"### {variant}")
            lines.append("")
            lines.append(f"- Runs: {aggregate['run_count']}")
            for metric, values in aggregate["metrics"].items():
                lines.append(f"- `{metric}`: {values['mean']:.6f} ± {values['std']:.6f}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def apply_route_parameters(service: Any, parameters: Mapping[str, Any]) -> None:
    """以白名单方式把消融参数应用到研究路线实例。"""

    targets = {
        "graph.seed_count": (service.graph_route, "seed_count"),
        "graph.max_edges": (service.graph_route, "max_edges"),
        "multimodal.requested_boost": (service.multimodal_route, "requested_boost"),
        "multimodal.secondary_boost": (service.multimodal_route, "secondary_boost"),
        "agent.max_steps": (service.agent_route, "max_steps"),
    }
    unknown = sorted(set(parameters) - set(targets))
    if unknown:
        raise ValueError(f"Unsupported research parameters: {', '.join(unknown)}")
    for key, value in parameters.items():
        target, attribute = targets[key]
        setattr(target, attribute, value)
