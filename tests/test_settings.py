"""验证配置定义唯一且 YAML 不允许重复键。"""

from __future__ import annotations

from pathlib import Path

import pytest

from coal_kb.infra.config import loader as config_loader
from coal_kb.infra.config import models as config_models


def test_settings_class_definitions_unique() -> None:
    text = Path(config_models.__file__).read_text(encoding="utf-8")
    for name in (
        "RegistryConfig",
        "ModelVersionsConfig",
        "ElasticConfig",
        "QueryRewriteConfig",
    ):
        assert text.count(f"class {name}") == 1


def test_yaml_duplicate_keys_raise(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "backend: elastic\nbackend: chroma\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate key"):
        config_loader._load_yaml_unique_keys(config_path)
