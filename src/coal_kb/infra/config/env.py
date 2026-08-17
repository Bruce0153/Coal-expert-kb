from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Stable environment-variable overrides shared by local and production runs."""

    model_config = SettingsConfigDict(env_prefix="COAL_KB_", extra="ignore")

    config: str = "configs/app.yaml"
    data_root: Optional[str] = None
    elastic_url: Optional[str] = None

    chroma_dir: Optional[str] = None
    sqlite_path: Optional[str] = None
    log_level: Optional[str] = None

    llm_model: Optional[str] = None
    embeddings_model: Optional[str] = None
