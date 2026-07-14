"""检查核心模块依赖方向，防止 canonical 层反向依赖已删除模块。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def _read_package(name: str) -> str:
    package = ROOT / name
    return "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))


def test_answering_layer_does_not_depend_on_interfaces() -> None:
    text = _read_package("answering")
    assert "coal_kb.interfaces" not in text
    assert "fastapi" not in text


def test_retrieval_layer_does_not_depend_on_answering_or_interfaces() -> None:
    text = _read_package("retrieval")
    assert "coal_kb.answering" not in text
    assert "coal_kb.interfaces" not in text


def test_context_layer_does_not_depend_on_answering() -> None:
    text = _read_package("context")
    assert "coal_kb.answering" not in text
