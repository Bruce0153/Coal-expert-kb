from __future__ import annotations

"""Composable ingestion stages for modern pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document

from coal_kb.chunking.splitter import split_docs_markdown_hierarchical_semantic
from coal_kb.loaders import load_any
from coal_kb.loaders.pdf_loader import PDFLoader


@dataclass
class SourceScanner:
    include_exts: set[str]
    exclude_exts: set[str]

    def scan(self, dirs: Iterable[Path]) -> Dict[str, Path]:
        files: Dict[str, Path] = {}
        for base_dir in dirs:
            if not base_dir.exists():
                continue
            for path in sorted(base_dir.rglob("*")):
                if not path.is_file():
                    continue
                ext = path.suffix.lower().lstrip(".")
                if ext in self.exclude_exts:
                    continue
                if ext and ext in self.include_exts:
                    files[str(path.resolve())] = path
        return files


@dataclass
class DocumentLoader:
    pdf_loader: PDFLoader

    def load(self, path: Path) -> List[Document]:
        if path.suffix.lower() == ".pdf":
            return self.pdf_loader.load(str(path))
        return load_any(str(path))


@dataclass
class Chunker:
    chunk_cfg: object

    def split(self, docs: List[Document]) -> List[Document]:
        return split_docs_markdown_hierarchical_semantic(docs, self.chunk_cfg.model_dump() if hasattr(self.chunk_cfg, "model_dump") else self.chunk_cfg)
