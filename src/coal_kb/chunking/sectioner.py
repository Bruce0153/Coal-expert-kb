from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Heuristic section detection for scientific papers (works for EN/CN mixed).
SECTION_PATTERNS = [
    ("abstract", r"\babstract\b|摘要"),
    ("introduction", r"\bintroduction\b|引言|绪论"),
    ("methods", r"\bexperimental\b|\bmaterials?\b|\bmethods?\b|实验|方法"),
    ("conditions", r"\bconditions?\b|\boperating\b|工况|操作条件|实验条件"),
    ("results", r"\bresults?\b|结果"),
    ("discussion", r"\bdiscussion\b|讨论"),
    ("conclusion", r"\bconclusions?\b|结论"),
    ("references", r"\breferences?\b|参考文献|文献"),
    ("acknowledgements", r"\backnowledg(e)?ments?\b|致谢"),
    ("appendix", r"\bappendix\b|附录"),
    ("contents", r"\btable\s+of\s+contents\b|目录"),
    ("table", r"\btable\b|表\s*\d+"),
    ("figure", r"\bfigure\b|图\s*\d+"),
]

_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in SECTION_PATTERNS]
_RX_DOI = re.compile(r"\bdoi\b|\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_RX_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RX_CITATION_NUM = re.compile(r"^\s*\d+\.\s+")
_RX_CITATION_NAME = re.compile(r"^[A-Z][a-z]+\s+[A-Z]\.")
_RX_JOURNAL = re.compile(r"\b(J\.|Vol\.|pp\.|Fuel|Energy\s*&\s*Fuels|Combustion|Proceedings)\b", re.I)

def _reference_features(text: str) -> Dict[str, int]:
    head = text[:1200]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    doi_count = len(_RX_DOI.findall(head))
    year_count = len(_RX_YEAR.findall(head))
    journal_hits = len(_RX_JOURNAL.findall(head))
    citation_line_count = sum(
        1 for ln in lines if _RX_CITATION_NUM.search(ln) or _RX_CITATION_NAME.search(ln)
    )
    return {
        "doi_count": doi_count,
        "year_count": year_count,
        "journal_hits": journal_hits,
        "citation_line_count": citation_line_count,
        "line_count": len(lines),
    }


def is_reference_like(text: str) -> bool:
    feats = _reference_features(text)
    if feats["citation_line_count"] >= 4:
        return True
    if feats["doi_count"] >= 1 and feats["year_count"] >= 3:
        return True
    if feats["journal_hits"] >= 3 and feats["year_count"] >= 4:
        return True
    if feats["year_count"] >= 8 and feats["citation_line_count"] >= 2:
        return True
    return False


def infer_section_with_debug(text: str) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Infer a coarse section label from text content.
    Returns (section, debug_features).
    """
    head = text[:1200]
    feats = _reference_features(text)

    if is_reference_like(text):
        return "references", feats

    for name, rx in _COMPILED:
        if rx.search(head):
            return name, feats
    return None, feats


def infer_section(text: str) -> Optional[str]:
    section, _ = infer_section_with_debug(text)
    return section
