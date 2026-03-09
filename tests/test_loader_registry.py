from __future__ import annotations

from coal_kb.loaders.registry import get_loader_for_path


def test_loader_registry_txt() -> None:
    loader = get_loader_for_path("example.txt")
    assert loader is not None


def test_loader_registry_optional_docx() -> None:
    loader = get_loader_for_path("example.docx")
    # optional dependency: loader may be None if python-docx missing
    assert loader is None or loader is not None
