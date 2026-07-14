"""保留旧 store 包入口，推荐改用 infra.persistence 与 indexing。"""

from coal_kb.infra.persistence.registry import Registry, RegistrySQLite
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.persistence.sql import SQLiteStore
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.indexing.manifest import Manifest, ManifestEntry
from coal_kb.indexing.validation import validate_index

__all__ = [
    "ChromaStore",
    "ElasticStore",
    "Manifest",
    "ManifestEntry",
    "Registry",
    "RegistrySQLite",
    "SQLiteStore",
    "validate_index",
]
