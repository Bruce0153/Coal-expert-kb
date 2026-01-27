from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set, Tuple


_WS_RE = re.compile(r"[ \t\r\f\v]+")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    cleaned = "\n".join([ln for ln in lines if ln != ""])
    return cleaned.strip()


def repair_hyphenation(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def basic_clean(text: str) -> str:
    return normalize_whitespace(repair_hyphenation(text))


def _norm_line(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def find_common_header_footer_lines(
    page_texts: List[str],
    *,
    head_n: int = 3,
    tail_n: int = 3,
    min_ratio: float = 0.6,
    min_len: int = 6,
    max_len: int = 120,
) -> Set[str]:
    """
    Detect common header/footer lines by counting only first/last N lines of each page.
    """
    if not page_texts:
        return set()

    cnt = Counter()
    total_pages = len(page_texts)

    for text in page_texts:
        lines = [_norm_line(x) for x in text.split("\n") if _norm_line(x)]
        heads = lines[:head_n]
        tails = lines[-tail_n:] if tail_n > 0 else []
        for ln in heads + tails:
            if len(ln) < min_len or len(ln) > max_len:
                continue
            cnt[ln] += 1

    common = set()
    for ln, c in cnt.items():
        if c >= 2 and (c / total_pages) >= min_ratio:
            common.add(ln)
    return common


def remove_common_header_footer(
    text: str,
    common_lines: Set[str],
    *,
    head_n: int = 3,
    tail_n: int = 3,
) -> str:
    """
    Remove common lines only if they appear in header/footer positions.
    """
    if not common_lines:
        return text

    lines = [_norm_line(x) for x in text.split("\n") if _norm_line(x)]
    if not lines:
        return text

    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n > 0 else []

    def keep_line(i: int, ln: str) -> bool:
        # only drop if it is in head/tail region
        if (i < head_n) and (ln in common_lines):
            return False
        if (i >= len(lines) - tail_n) and (ln in common_lines):
            return False
        return True

    kept = [ln for i, ln in enumerate(lines) if keep_line(i, ln)]
    return "\n".join(kept).strip()
