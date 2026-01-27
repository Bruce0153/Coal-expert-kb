from __future__ import annotations

import re
from typing import Optional

# Heuristic section detection for scientific papers (works for EN/CN mixed).
SECTION_PATTERNS = [
    ("abstract", r"\babstract\b|摘要"),
    ("introduction", r"\bintroduction\b|引言|绪论"),
    ("methods", r"\bexperimental\b|\bmaterials?\b|\bmethods?\b|实验|方法"),
    ("conditions", r"\bconditions?\b|\boperating\b|工况|操作条件|实验条件"),
    ("results", r"\bresults?\b|结果"),
    ("discussion", r"\bdiscussion\b|讨论"),
    ("conclusion", r"\bconclusions?\b|结论"),
    ("references", r"\breferences?\b|参考文献"),
    ("table", r"\btable\b|表\s*\d+"),
    ("figure", r"\bfigure\b|图\s*\d+"),
]

_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in SECTION_PATTERNS]


def infer_section(text: str) -> Optional[str]:
    """
    Infer a coarse section label from text content.
    This is heuristic: good enough for retrieval boosting/metadata.
    """
    head = text[:800]  # look at first chunk of the page
    for name, rx in _COMPILED:
        if rx.search(head):
            return name
    return None
