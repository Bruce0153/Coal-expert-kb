from __future__ import annotations

from pathlib import Path

import pytest

from coal_kb import settings


def test_settings_class_definitions_unique() -> None:
    text = Path(settings.__file__).read_text(encoding="utf-8")
    for name in ("RegistryConfig", "ModelVersionsConfig", "ElasticConfig", "QueryRewriteConfig"):
        assert text.count(f"class {name}") == 1


def test_yaml_duplicate_keys_raise(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("backend: elastic\nbackend: chroma\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key"):
        settings._load_yaml_unique_keys(config_path)
