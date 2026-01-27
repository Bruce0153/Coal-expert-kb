from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class TableExtractor:
    """
    Optional table extraction from PDFs.

    Strategy:
    - If Camelot is installed and PDF is parseable: extract tables and return as Documents.
    - Otherwise: return empty list, and rely on standard text extraction.

    Why optional?
    - Table extraction quality is highly dependent on PDF structure.
    - Camelot requires additional system deps and works best for vector PDFs.
    """

    flavor: str = "lattice"  # or "stream"

    def extract(self, pdf_path: Path) -> List[Document]:
        try:
            import camelot  # type: ignore
        except Exception:
            logger.info("Camelot not installed; skip table extraction.")
            return []

        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=self.flavor)
        except Exception as e:
            logger.warning("Camelot failed on %s: %s", pdf_path, e)
            return []

        docs: List[Document] = []
        for i, t in enumerate(tables):
            try:
                df = t.df
                text = df.to_csv(index=False)
                meta = {
                    "source_file": str(pdf_path),
                    "section": "table",
                    "table_index": i,
                    "table_page": getattr(t, "page", None),
                    "table_flavor": self.flavor,
                }
                docs.append(Document(page_content=text, metadata=meta))
            except Exception as e:
                logger.debug("Failed to convert table %s #%d: %s", pdf_path, i, e)

        logger.info("Extracted %d tables from %s", len(docs), pdf_path)
        return docs
