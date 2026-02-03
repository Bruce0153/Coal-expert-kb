from __future__ import annotations

import re
from typing import List

_SPLIT_RE = re.compile(r"(?<=[\.\!\?\。\！？；;:：])\s*")


def split_sentences(text: str, *, min_len: int = 8) -> List[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return []
    merged: List[str] = []
    buffer = ""
    for part in parts:
        if not buffer:
            buffer = part
            continue
        if len(buffer) < min_len:
            buffer += part
        else:
            merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)
    return merged
