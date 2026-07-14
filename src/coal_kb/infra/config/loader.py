from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """
    Load YAML config + .env overrides.
    """
    load_dotenv(override=False)
    env = EnvSettings()

    config_path = Path(env.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Set COAL_KB_CONFIG or create configs/app.yaml."
        )

    raw = _load_yaml_unique_keys(config_path)
    cfg = AppConfig.model_validate(raw)

    # Apply env overrides (keep minimal)
    if env.chroma_dir:
        cfg.paths.chroma_dir = env.chroma_dir
    if env.sqlite_path:
        cfg.paths.sqlite_path = env.sqlite_path
    if env.log_level:
        cfg.logging.level = env.log_level

    if env.llm_model:
        cfg.llm.model = env.llm_model
    if env.embeddings_model:
        cfg.embeddings.model = env.embeddings_model

    return _ensure_dirs(cfg)

def _load_yaml_unique_keys(path: Path) -> dict:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict:
        mapping = {}
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
