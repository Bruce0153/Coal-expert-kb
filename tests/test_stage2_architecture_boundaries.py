"""检查第二阶段模块依赖方向，防止 canonical 层反向依赖旧 facade。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def _python_texts(relative_dir: str) -> list[str]:
    return [path.read_text(encoding="utf-8") for path in (ROOT / relative_dir).rglob("*.py")]


def test_recall_layer_does_not_depend_on_retrieval_or_interfaces() -> None:
    text = "\n".join(_python_texts("recall"))
    assert "coal_kb.retrieval" not in text
    assert "coal_kb.api" not in text
    assert "coal_kb.web" not in text


def test_answering_layer_does_not_depend_on_legacy_generation_or_interfaces() -> None:
    text = "\n".join(_python_texts("answering"))
    assert "coal_kb.generation" not in text
    assert "coal_kb.api" not in text
    assert "coal_kb.web" not in text


def test_context_layer_does_not_depend_on_answering_or_interfaces() -> None:
    text = "\n".join(_python_texts("context"))
    assert "coal_kb.answering" not in text
    assert "coal_kb.api" not in text
    assert "coal_kb.web" not in text
