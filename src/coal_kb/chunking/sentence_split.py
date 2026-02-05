from __future__ import annotations

import re
from typing import List

_SPLIT_RE = re.compile(r"(?<=[\.!?。！？；;:：])\s*")
_ABBREV = {
    "e.g.",
    "i.e.",
    "etc.",
    "dr.",
    "mr.",
    "mrs.",
    "vs.",
    "fig.",
}


def split_sentences(text: str, *, min_len: int = 8) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text.strip()) if p.strip()]
    if not parts:
        return []

    merged_abbrev: List[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        lower = cur.lower()
        if any(lower.endswith(abbrev) for abbrev in _ABBREV) and i + 1 < len(parts):
            merged_abbrev.append(f"{cur} {parts[i + 1]}")
            i += 2
            continue
        merged_abbrev.append(cur)
        i += 1

    merged: List[str] = []
    buffer = ""
    for part in merged_abbrev:
        if not buffer:
            buffer = part
            continue
        if len(buffer) < min_len:
            joiner = "" if re.search(r"[\u4e00-\u9fff]$", buffer) else " "
            buffer = f"{buffer}{joiner}{part}"
        else:
            merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)
    return merged
