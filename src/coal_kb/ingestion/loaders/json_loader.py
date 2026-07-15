"""加载 JSON 与 JSONL 文档。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.ingestion.loaders.base import BaseLoader, detect_language, normalize_text
from coal_kb.ingestion.loaders.registry import register_loader
from coal_kb.utils.jsonl import iter_jsonl


class JSONLoader(BaseLoader):
    def load(self, path: str) -> list[Document]:
        source = Path(path)
        if source.suffix.lower() == ".jsonl":
            return [
                self._document(path, payload, row_number - 1)
                for row_number, payload in iter_jsonl(source)
            ]

        payload = json.loads(
            source.read_text(encoding="utf-8", errors="ignore")
        )
        values = payload if isinstance(payload, list) else [payload]
        return [
            self._document(path, value, index)
            for index, value in enumerate(values)
        ]

    @staticmethod
    def _document(path: str, payload: Any, index: int) -> Document:
        content = normalize_text(json.dumps(payload, ensure_ascii=False))
        return Document(
            page_content=content,
            metadata={
                "source_file": path,
                "record_id": index,
                "section": "record",
                "doc_type": "jsonl" if path.endswith(".jsonl") else "json",
                "language": detect_language(content),
                "parser": "json",
            },
        )


register_loader("json", JSONLoader)
register_loader("jsonl", JSONLoader)
