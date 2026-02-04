# src/coal_kb/llm/enrich_cache.py
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_cache_key(
    *,
    chunk_id: str,
    model: str,
    prompt_version: str,
    text: str,
) -> str:
    """
    Key is stable across runs:
    - chunk_id identifies the chunk
    - model + prompt_version ensure changes invalidate cache
    - text hash ensures if chunk text changes we re-enrich
    """
    th = _sha1(text or "")
    raw = f"{chunk_id}||{model}||{prompt_version}||{th}"
    return _sha1(raw)


@dataclass
class EnrichCache:
    sqlite_path: str

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.sqlite_path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_enrich_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    text_sha1 TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_enrich_chunk ON llm_enrich_cache(chunk_id);"
            )

    def get_many(self, keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        keys_list = list(keys)
        if not keys_list:
            return {}
        placeholders = ",".join(["?"] * len(keys_list))
        out: Dict[str, Dict[str, Any]] = {}
        with self._connect() as con:
            cur = con.execute(
                f"SELECT cache_key, payload_json FROM llm_enrich_cache WHERE cache_key IN ({placeholders})",
                keys_list,
            )
            for k, payload_json in cur.fetchall():
                try:
                    out[k] = json.loads(payload_json)
                except Exception:
                    # corrupted payload shouldn't crash query
                    continue
        return out

    def upsert_many(
        self,
        rows: List[Tuple[str, str, str, str, str, Dict[str, Any]]],
    ) -> None:
        """
        rows: (cache_key, chunk_id, model, prompt_version, text_sha1, payload_dict)
        """
        if not rows:
            return
        now = int(time.time())
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO llm_enrich_cache(
                    cache_key, chunk_id, model, prompt_version, text_sha1, created_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at
                """,
                [
                    (ck, cid, model, pv, tsha1, now, json.dumps(payload, ensure_ascii=False))
                    for (ck, cid, model, pv, tsha1, payload) in rows
                ],
            )


def chunk_id_for_doc(d: Document, fallback_prefix: str = "chunk") -> str:
    """
    Prefer existing chunk_id. If absent, derive a stable id from (source_file,page,hash(text)).
    """
    m = d.metadata or {}
    cid = m.get("chunk_id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()

    source = str(m.get("source_file") or m.get("source") or "unknown")
    page = str(m.get("page") or m.get("page_number") or "0")
    text = d.page_content or ""
    return f"{fallback_prefix}:{_sha1(source)}:{page}:{_sha1(text)[:12]}"
