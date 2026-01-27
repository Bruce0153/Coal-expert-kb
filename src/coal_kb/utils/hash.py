from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_chunk_id(*parts: str) -> str:
    """
    Generate a stable chunk_id from a list of strings.
    """
    joined = "||".join(parts)
    return sha256_text(joined)[:24]
