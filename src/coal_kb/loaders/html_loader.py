from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language, normalize_text
from coal_kb.loaders.registry import register_loader


class HTMLLoader(BaseLoader):
    def __init__(self) -> None:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("beautifulsoup4 is required for HTMLLoader") from exc

    def load(self, path: str) -> List[Document]:
        from bs4 import BeautifulSoup

        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = normalize_text(soup.get_text(" "))
        lang = detect_language(text)
        return [
            Document(
                page_content=text,
                metadata={
                    "source_file": path,
                    "section": "body",
                    "title": title,
                    "doc_type": "html",
                    "language": lang,
                    "parser": "html",
                },
            )
        ]


register_loader("html", HTMLLoader)
