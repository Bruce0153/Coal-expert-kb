"""验证索引与持久化新路径、旧兼容路径及依赖方向。"""

from __future__ import annotations

import ast
from pathlib import Path

from coal_kb.indexing.manifest import Manifest as CanonicalManifest
from coal_kb.indexing.validation import validate_index as canonical_validate_index
from coal_kb.infra.persistence.registry.sqlite import RegistrySQLite as CanonicalRegistrySQLite
from coal_kb.infra.persistence.search.elasticsearch import ElasticStore as CanonicalElasticStore
from coal_kb.infra.persistence.sql.records import SQLiteStore as CanonicalSQLiteStore
from coal_kb.infra.persistence.vector.chroma import ChromaStore as CanonicalChromaStore
from coal_kb.store.chroma_store import ChromaStore as LegacyChromaStore
from coal_kb.store.elastic_store import ElasticStore as LegacyElasticStore
from coal_kb.store.elastic_validation import validate_index as legacy_validate_index
from coal_kb.store.manifest import Manifest as LegacyManifest
from coal_kb.store.registry_sqlite import RegistrySQLite as LegacyRegistrySQLite
from coal_kb.store.sql_store import SQLiteStore as LegacySQLiteStore


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_legacy_store_exports_point_to_canonical_objects() -> None:
    assert LegacyChromaStore is CanonicalChromaStore
    assert LegacyElasticStore is CanonicalElasticStore
    assert LegacyRegistrySQLite is CanonicalRegistrySQLite
    assert LegacySQLiteStore is CanonicalSQLiteStore
    assert LegacyManifest is CanonicalManifest
    assert legacy_validate_index is canonical_validate_index


def test_active_code_does_not_import_legacy_store_modules() -> None:
    root = Path("src/coal_kb")
    compatibility_root = root / "store"
    forbidden_prefix = "coal_kb.store"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.is_relative_to(compatibility_root):
            continue
        for module in _imports(path):
            if module.startswith(forbidden_prefix):
                violations.append(f"{path}: {module}")
    assert not violations, "Legacy store imports remain:\n" + "\n".join(violations)


def test_indexing_package_uses_lazy_exports() -> None:
    text = Path("src/coal_kb/indexing/__init__.py").read_text(encoding="utf-8")
    assert "def __getattr__" in text
    assert "from .service import IndexService" not in text.split("def __getattr__", 1)[0]
