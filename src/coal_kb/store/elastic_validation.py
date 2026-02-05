from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings

logger = logging.getLogger(__name__)


def resolve_index_name(client: Any, index_or_alias: str) -> str:
    if client.indices.exists(index=index_or_alias):
        return index_or_alias
    if client.indices.exists_alias(name=index_or_alias):
        data = client.indices.get_alias(name=index_or_alias)
        return next(iter(data.keys()))
    raise ValueError(f"Index or alias not found: {index_or_alias}")


def validate_index(
    *,
    client: Any,
    index_or_alias: str,
    embeddings_cfg: EmbeddingsConfig,
    expected_dims: Optional[int] = None,
    query_text: str = "validation probe",
    min_docs: int = 1,
) -> Dict[str, object]:
    index_name = resolve_index_name(client, index_or_alias)
    errors: list[str] = []

    count_rsp = client.count(index=index_name)
    doc_count = int(count_rsp.get("count", 0))
    if doc_count < min_docs:
        errors.append(f"doc_count={doc_count} < {min_docs}")

    mapping = client.indices.get_mapping(index=index_name)
    props = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
    embedding_props = props.get("embedding") or {}
    dims = embedding_props.get("dims")

    required_fields = {
        "is_parent": "boolean",
        "parent_id": "keyword",
        "heading_path": "keyword",
        "chunk_level": "short",
    }
    for fname, ftype in required_fields.items():
        node = props.get(fname)
        if not node:
            errors.append(f"missing field: {fname}")
            continue
        if node.get("type") != ftype:
            errors.append(f"field type mismatch: {fname}={node.get('type')} expected={ftype}")

    expected = expected_dims or embeddings_cfg.dimensions
    if expected is None:
        embeddings = make_embeddings(embeddings_cfg)
        expected = len(embeddings.embed_query(query_text))
    if dims != expected:
        errors.append(f"embedding dims mismatch: mapping={dims} expected={expected}")

    embeddings = make_embeddings(embeddings_cfg)
    query_vec = embeddings.embed_query(query_text)
    try:
        rsp = client.search(
            index=index_name,
            body={
                "size": 1,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vec,
                    "k": 1,
                    "num_candidates": 10,
                },
            },
        )
        hits = rsp.get("hits", {}).get("hits", [])
        if not hits:
            errors.append("knn self-check returned no hits")
    except Exception as exc:
        errors.append(f"knn self-check failed: {exc}")

    return {
        "index_name": index_name,
        "doc_count": doc_count,
        "embedding_dims": dims,
        "expected_dims": expected,
        "ok": not errors,
        "errors": errors,
    }
