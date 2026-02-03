from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language, normalize_text
from coal_kb.loaders.registry import register_loader


class TextLoader(BaseLoader):
    def load(self, path: str) -> List[Document]:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        docs: List[Document] = []
        paragraph = []
        start = 1
        line_no = 0
        current_title = None
        for line in text.splitlines():
            line_no += 1
            stripped = line.strip()
            if stripped.startswith("#"):
                current_title = stripped.lstrip("#").strip()
            if stripped:
                paragraph.append(stripped)
                continue
            if paragraph:
                content = normalize_text(" ".join(paragraph))
                lang = detect_language(content)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_file": path,
                            "line_range": [start, line_no],
                            "section": "body",
                            "title": current_title,
                            "doc_type": Path(path).suffix.lstrip("."),
                            "language": lang,
                            "parser": "text",
                        },
                    )
                )
                paragraph = []
                start = line_no + 1
        if paragraph:
            content = normalize_text(" ".join(paragraph))
            lang = detect_language(content)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_file": path,
                        "line_range": [start, line_no],
                        "section": "body",
                        "title": current_title,
                        "doc_type": Path(path).suffix.lstrip("."),
                        "language": lang,
                        "parser": "text",
                    },
                )
            )
        return docs


register_loader("txt", TextLoader)
register_loader("md", TextLoader)
