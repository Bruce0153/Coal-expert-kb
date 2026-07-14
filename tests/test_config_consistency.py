"""验证 embedding 配置只有一个正式入口。"""

from pathlib import Path

from coal_kb.infra.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_embedding_config_has_single_public_entry() -> None:
    cfg = AppConfig()
    assert hasattr(cfg, "embeddings")
    assert not hasattr(cfg, "embedding")
    assert cfg.embeddings.model


def test_yaml_does_not_define_duplicate_embedding_section() -> None:
    text = (ROOT / "configs/app.yaml").read_text(encoding="utf-8")
    assert "\nembedding:\n" not in text
    assert text.count("\nembeddings:\n") == 1


def test_environment_override_uses_embeddings_model(monkeypatch) -> None:
    monkeypatch.setenv("COAL_KB_EMBEDDINGS_MODEL", "test-embedding-model")
    load_config.cache_clear()
    cfg = load_config()
    assert cfg.embeddings.model == "test-embedding-model"
    load_config.cache_clear()
