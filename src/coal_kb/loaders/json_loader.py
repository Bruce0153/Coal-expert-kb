from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language, normalize_text
from coal_kb.loaders.registry import register_loader


class JSONLoader(BaseLoader):
    def load(self, path: str) -> List[Document]:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        docs: List[Document] = []
        if path.endswith(".jsonl"):
            for idx, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append(self._doc_from_obj(path, obj, idx))
            return docs
        obj = json.loads(text)
        if isinstance(obj, list):
            for idx, rec in enumerate(obj):
                docs.append(self._doc_from_obj(path, rec, idx))
            return docs
        docs.append(self._doc_from_obj(path, obj, 0))
        return docs

    def _doc_from_obj(self, path: str, obj: Any, idx: int) -> Document:
        content = normalize_text(json.dumps(obj, ensure_ascii=False))
        lang = detect_language(content)
        return Document(
            page_content=content,
            metadata={
                "source_file": path,
                "record_id": idx,
                "section": "record",
                "doc_type": "jsonl" if path.endswith(".jsonl") else "json",
                "language": lang,
                "parser": "json",
            },
        )


register_loader("json", JSONLoader)
register_loader("jsonl", JSONLoader)
