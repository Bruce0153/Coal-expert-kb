from __future__ import annotations

import importlib.util
import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MISSING_DEPS = [
    name
    for name in ("yaml", "pydantic", "langchain_core", "langchain_chroma", "sqlalchemy")
    if importlib.util.find_spec(name) is None
]


def pytest_ignore_collect(collection_path, config):
    if MISSING_DEPS:
        return collection_path.name != "test_environment.py"
    return False
