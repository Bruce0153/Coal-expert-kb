from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ManifestEntry:
    path: str
    mtime_ns: int
    size: int
    sha256: str
    chunk_count: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "sha256": self.sha256,
            "chunk_count": self.chunk_count,
        }

    @staticmethod
    def from_dict(data: dict) -> "ManifestEntry":
        return ManifestEntry(
            path=str(data.get("path", "")),
            mtime_ns=int(data.get("mtime_ns", 0)),
            size=int(data.get("size", 0)),
            sha256=str(data.get("sha256", "")),
            chunk_count=int(data.get("chunk_count", 0)),
        )


@dataclass
class Manifest:
    version: int = 1
    embeddings_signature: str = ""
    chunking_signature: str = ""
    schema_signature: str = ""
    files: Dict[str, ManifestEntry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "embeddings_signature": self.embeddings_signature,
            "chunking_signature": self.chunking_signature,
            "schema_signature": self.schema_signature,
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> "Manifest":
        files = {
            k: ManifestEntry.from_dict(v)
            for k, v in (data.get("files") or {}).items()
            if isinstance(v, dict)
        }
        return Manifest(
            version=int(data.get("version", 1)),
            embeddings_signature=str(data.get("embeddings_signature", "")),
            chunking_signature=str(data.get("chunking_signature", "")),
            schema_signature=str(data.get("schema_signature", "")),
            files=files,
        )

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return Manifest()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Manifest.from_dict(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def signature_mismatch(self, *, embeddings: str, chunking: str, schema: str) -> Dict[str, str]:
        mismatch: Dict[str, str] = {}
        if self.embeddings_signature and self.embeddings_signature != embeddings:
            mismatch["embeddings"] = self.embeddings_signature
        if self.chunking_signature and self.chunking_signature != chunking:
            mismatch["chunking"] = self.chunking_signature
        if self.schema_signature and self.schema_signature != schema:
            mismatch["schema"] = self.schema_signature
        return mismatch
