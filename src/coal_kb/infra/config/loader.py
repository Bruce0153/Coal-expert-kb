from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .env import EnvSettings
from .models import AppConfig


def _ensure_dirs(cfg: AppConfig) -> AppConfig:
    Path(cfg.paths.raw_pdfs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.raw_docs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.interim_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.registry.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """加载支持单继承的 YAML 配置，避免生产配置复制整份 app.yaml。"""
    resolved = path.resolve()
    chain = set(seen or set())
    if resolved in chain:
        raise ValueError(f"Config inheritance cycle detected at {path}")
    chain.add(resolved)

    raw = _load_yaml_unique_keys(path)
    parent_ref = raw.pop("extends", None)
    if not parent_ref:
        return raw
    parent_path = Path(str(parent_ref))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent config not found: {parent_path}")
    return _deep_merge(_load_config_mapping(parent_path, chain), raw)


def _apply_data_root(cfg: AppConfig, data_root: str) -> None:
    root = Path(data_root)
    cfg.paths.raw_pdfs_dir = str(root / "raw_pdfs")
    cfg.paths.raw_docs_dir = str(root / "raw_docs")
    cfg.paths.interim_dir = str(root / "interim")
    cfg.paths.artifacts_dir = str(root / "artifacts")
    cfg.paths.chroma_dir = str(root / "chroma_db")
    cfg.paths.sqlite_path = str(root / "expert.db")
    cfg.paths.manifest_path = str(root / "manifest.json")
    cfg.registry.sqlite_path = str(root / "kb.db")
    cfg.complex_qa.table_records_path = str(root / "interim" / "table_records.jsonl")


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load YAML configuration and stable environment overrides."""
    load_dotenv(override=False)
    env = EnvSettings()

    config_path = Path(env.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Set COAL_KB_CONFIG or create configs/app.yaml."
        )

    cfg = AppConfig.model_validate(_load_config_mapping(config_path))

    if env.data_root:
        _apply_data_root(cfg, env.data_root)
    if env.elastic_url:
        cfg.elastic.host = env.elastic_url
    if env.chroma_dir:
        cfg.paths.chroma_dir = env.chroma_dir
    if env.sqlite_path:
        cfg.paths.sqlite_path = env.sqlite_path
    if env.log_level:
        cfg.logging.level = env.log_level
    if env.llm_model:
        cfg.llm.active.model = env.llm_model
    if env.embeddings_model:
        cfg.embeddings.active.model = env.embeddings_model

    return _ensure_dirs(cfg)


def _load_yaml_unique_keys(path: Path) -> dict[str, Any]:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key in YAML: {key}")
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )

    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader) or {}
    except ValueError as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc
