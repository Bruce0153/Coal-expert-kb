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

        logger.info("Table extraction start | pdf=%s flavor=%s", pdf_path, self.flavor)
        if self.flavor == "auto":
            tables = self._read_with_fallback(camelot, pdf_path)
        else:
            tables = self._read_single(camelot, pdf_path, self.flavor)
            if tables is None:
                logger.warning("Table extraction failed | pdf=%s flavor=%s", pdf_path, self.flavor)
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

        if not docs:
            logger.warning("No tables extracted | pdf=%s flavor=%s", pdf_path, self.flavor)
        else:
            logger.info("Extracted %d tables from %s", len(docs), pdf_path)
        return docs

    def _read_single(self, camelot: object, pdf_path: Path, flavor: str):
        try:
            return camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception as e:
            logger.warning("Camelot failed on %s with flavor=%s: %s", pdf_path, flavor, e)
            return None

    def _read_with_fallback(self, camelot: object, pdf_path: Path):
        tables = self._read_single(camelot, pdf_path, "lattice")
        if tables and len(tables) > 0:
            return tables
        if tables is not None:
            logger.warning("Camelot lattice extracted 0 tables; falling back to stream for %s", pdf_path)
        tables = self._read_single(camelot, pdf_path, "stream")
        return tables or []
