from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language, normalize_text
from coal_kb.loaders.registry import register_loader


class CSVLoader(BaseLoader):
    def load(self, path: str) -> List[Document]:
        docs: List[Document] = []
        with Path(path).open(encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        chunk = []
        start = 1
        for idx, row in enumerate(rows, start=1):
            chunk.append(", ".join(row))
            if len(chunk) >= 50:
                content = normalize_text("\n".join(chunk))
                lang = detect_language(content)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_file": path,
                            "row_range": [start, idx],
                            "section": "rows",
                            "doc_type": "csv",
                            "language": lang,
                            "parser": "csv",
                        },
                    )
                )
                chunk = []
                start = idx + 1
        if chunk:
            content = normalize_text("\n".join(chunk))
            lang = detect_language(content)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_file": path,
                        "row_range": [start, len(rows)],
                        "section": "rows",
                        "doc_type": "csv",
                        "language": lang,
                        "parser": "csv",
                    },
                )
            )
        return docs


register_loader("csv", CSVLoader)
