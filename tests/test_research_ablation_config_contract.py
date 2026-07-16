"""验证示例消融配置与脚本入口保持可运行契约。"""

from pathlib import Path

from coal_kb.research.suites import ExperimentSuiteConfig


def test_repository_ablation_config_is_versioned_and_expandable() -> None:
    config = ExperimentSuiteConfig.from_file(Path("configs/research/milestone_d_ablation.yaml"))

    assert config.version == "research-suite.v1"
    assert config.dataset_path.name == "evaluation_sample.jsonl"
    assert {variant.name for variant in config.variants} >= {
        "standard",
        "graph-default",
        "multimodal-default",
        "agent-default",
    }
    assert len(config.expand_runs()) == len(config.variants) * config.repeats
