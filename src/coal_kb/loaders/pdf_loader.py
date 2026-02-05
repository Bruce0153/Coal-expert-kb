from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language
from coal_kb.loaders.markdown_clean import (
    collapse_repeated_headers,
    fix_hyphenation,
    merge_wrapped_lines,
    normalize_bullets,
    normalize_whitespace,
)
from coal_kb.loaders.registry import register_loader
from coal_kb.parsing.pdf_loader import load_pdf_pages
from coal_kb.settings import PDFMarkdownConfig

logger = logging.getLogger(__name__)


@dataclass
class _LineSpan:
    x: float
    y: float
    text: str
    font_size: float


class PDFLoader(BaseLoader):
    def __init__(self, cfg: PDFMarkdownConfig | None = None) -> None:
        self.cfg = cfg or PDFMarkdownConfig()

    def _column_order(self, lines: List[_LineSpan]) -> List[_LineSpan]:
        if len(lines) <= 3 or self.cfg.two_column_mode == "off":
            return sorted(lines, key=lambda t: (t.y, t.x))
        xs = [l.x for l in lines]
        xmin, xmax = min(xs), max(xs)
        spread = xmax - xmin
        is_two_col = self.cfg.two_column_mode == "on" or spread > 180
        if not is_two_col:
            return sorted(lines, key=lambda t: (t.y, t.x))
        mid = (xmin + xmax) / 2
        left = sorted([l for l in lines if l.x <= mid], key=lambda t: (t.y, t.x))
        right = sorted([l for l in lines if l.x > mid], key=lambda t: (t.y, t.x))
        return left + right

    def _font_stats(self, page_dicts: List[Dict]) -> Tuple[float, float]:
        sizes: List[float] = []
        for d in page_dicts:
            for b in d.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        sz = float(span.get("size", 0) or 0)
                        if sz > 0:
                            sizes.append(sz)
        if not sizes:
            return 10.0, 12.0
        body = statistics.median(sizes)
        p85 = statistics.quantiles(sizes, n=100)[84] if len(sizes) >= 100 else max(sizes)
        return body, p85

    def _heading_level(self, font_size: float, body_font: float) -> int | None:
        if font_size < body_font * self.cfg.min_heading_font_ratio:
            return None
        ratio = font_size / max(body_font, 1.0)
        if ratio >= 1.55:
            return 1
        if ratio >= 1.35:
            return 2
        if ratio >= 1.2:
            return 3
        return 4

    def extract_pdf_markdown(self, pdf_path: Path) -> List[Document]:
        if not self.cfg.enabled:
            return []
        import fitz

        pdf = fitz.open(pdf_path)
        page_dicts: List[Dict] = []
        for page in pdf:
            page_dicts.append(page.get_text("dict"))
        pdf.close()
        if not page_dicts:
            return []

        body_font, _ = self._font_stats(page_dicts)
        page_markdowns: List[str] = []

        for pd in page_dicts:
            lines: List[_LineSpan] = []
            for b in pd.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for ln in b.get("lines", []):
                    spans = ln.get("spans", [])
                    txt = "".join(str(sp.get("text", "")) for sp in spans).strip()
                    if not txt:
                        continue
                    font_size = max(float(sp.get("size", 0) or 0) for sp in spans) if spans else body_font
                    bbox = ln.get("bbox") or [0, 0, 0, 0]
                    lines.append(_LineSpan(x=float(bbox[0]), y=float(bbox[1]), text=txt, font_size=font_size))

            if not lines:
                continue

            ordered = self._column_order(lines)
            md_lines: List[str] = []
            for ln in ordered:
                lvl = self._heading_level(ln.font_size, body_font)
                text = ln.text
                if lvl is not None:
                    depth = min(max(1, lvl), self.cfg.heading_max_depth)
                    md_lines.append(f"{'#' * depth} {text}")
                elif text.startswith(("•", "·", "- ", "* ")):
                    md_lines.append("- " + text.lstrip("•·-* "))
                elif text[:2].isdigit() and (text[2:3] in {".", ")"}):
                    md_lines.append(text[0:2] + ". " + text[3:].strip())
                else:
                    md_lines.append(text)

            page_md = "\n".join(md_lines)
            page_md = fix_hyphenation(page_md)
            page_md = merge_wrapped_lines(page_md)
            page_md = normalize_bullets(page_md)
            page_md = normalize_whitespace(page_md)
            page_markdowns.append(page_md)

        if self.cfg.drop_headers_footers and page_markdowns:
            page_markdowns = collapse_repeated_headers(page_markdowns)

        page_markdowns = [p for p in page_markdowns if p.strip()]
        if not page_markdowns:
            return []

        content = "\n\n".join(page_markdowns)
        meta = {
            "source_file": str(pdf_path),
            "doc_type": "pdf",
            "parser": "pymupdf_dict_markdown",
            "format": "markdown",
            "page_range": [1, len(page_markdowns)],
            "pages": [{"page": i + 1} for i in range(len(page_markdowns))],
            "language": detect_language(content),
        }
        return [Document(page_content=content, metadata=meta)]

    def load(self, path: str) -> List[Document]:
        p = Path(path)
        try:
            docs = self.extract_pdf_markdown(p)
            if docs:
                return docs
        except Exception as exc:
            logger.warning("PDF markdown extraction failed, fallback to text: %s", exc)

        docs = load_pdf_pages(p)
        for doc in docs:
            lang = detect_language(doc.page_content or "")
            doc.metadata = {
                **(doc.metadata or {}),
                "doc_type": "pdf",
                "language": lang,
                "parser": "pdf",
                "format": "text",
            }
        return docs


register_loader("pdf", PDFLoader)
