from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from coal_kb.chunking.sectioner import is_reference_like
from coal_kb.cli_ui import print_banner, print_stats_table
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.metadata.normalize import Ontology
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory


@dataclass
class EvalItem:
    query: str
    expected_sources: List[Dict[str, Any]]
    expected_stage: Optional[str] = None


def load_eval_set(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(
            EvalItem(
                query=obj["query"],
                expected_sources=obj.get("expected_sources") or [],
                expected_stage=obj.get("expected_stage"),
            )
        )
    return items


def match_gold(gold: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    chunk_id = gold.get("chunk_id")
    if chunk_id and str(meta.get("chunk_id")) == str(chunk_id):
        return True
    src = str(meta.get("source_file", "")).lower()
    page = gold.get("page", None)
    source_file = str(gold.get("source_file", "")).lower()
    if source_file and source_file not in src:
        return False
    if page is not None:
        return meta.get("page") == page
    return bool(source_file)


def range_overlap(meta: Dict[str, Any], query_range: Optional[List[float]], *, key_point: str, key_min: str, key_max: str) -> bool:
    if query_range is None:
        return True
    qlo, qhi = float(query_range[0]), float(query_range[1])
    dmin = meta.get(key_min)
    dmax = meta.get(key_max)
    if dmin is not None and dmax is not None:
        return max(float(dmin), qlo) <= min(float(dmax), qhi)
    x = meta.get(key_point)
    if x is None:
        return False
    return qlo <= float(x) <= qhi


def filter_match(meta: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    stage = parsed.get("stage")
    if stage and stage != "unknown":
        checks["stage"] = str(meta.get("stage")) == stage

    gas = parsed.get("gas_agent") or []
    if gas:
        checks["gas_agent"] = any(meta.get(f"gas_{g}") for g in gas)

    targets = parsed.get("targets") or []
    if targets:
        checks["targets"] = any(meta.get(f"has_{t}") for t in targets)

    T_range = parsed.get("T_range_K")
    if T_range:
        checks["T_range_K"] = range_overlap(meta, T_range, key_point="T_K", key_min="T_min_K", key_max="T_max_K")

    P_range = parsed.get("P_range_MPa")
    if P_range:
        checks["P_range_MPa"] = range_overlap(meta, P_range, key_point="P_MPa", key_min="P_min_MPa", key_max="P_max_MPa")

    return checks


def recall_at_k(docs: List[Dict[str, Any]], gold_sources: List[Dict[str, Any]], k: int) -> bool:
    for d in docs[:k]:
        meta = d
        if any(match_gold(g, meta) for g in gold_sources):
            return True
    return False


def _filter_precision_at_k(docs: List[Dict[str, Any]], parsed: Dict[str, Any], k: int) -> float:
    if not docs:
        return 0.0
    hits = 0
    for meta in docs[:k]:
        checks = filter_match(meta, parsed)
        if not checks:
            continue
        if all(checks.values()):
            hits += 1
    return hits / min(k, len(docs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality metrics.")
    parser.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    print_banner("Coal KB Retrieval Eval", f"backend={cfg.backend}")
    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)

    backend = cfg.backend
    if backend not in {"elastic", "chroma", "both"}:
        raise ValueError(f"Unsupported backend: {backend}")

    vector_factory = None
    if backend in {"chroma", "both"}:
        store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            embedding_model=cfg.embedding.model_name,
        )
        vector_factory = store.as_retriever

    if backend in {"elastic", "both"}:
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        vector_factory = make_elastic_retriever_factory(
            client=elastic_store.client,
            index=cfg.elastic.alias_current,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
        )

    expert = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=args.k,
        rerank_enabled=cfg.retrieval.rerank_enabled,
        rerank_model=cfg.retrieval.rerank_model,
        rerank_top_n=cfg.retrieval.rerank_top_n,
        rerank_candidates=cfg.retrieval.candidates,
        rerank_device=cfg.retrieval.rerank_device,
        max_per_source=cfg.retrieval.max_per_source,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=(backend != "elastic"),
        where_full=(backend == "elastic"),
    )

    items = load_eval_set(Path(args.gold))

    recalls = {1: 0, 3: 0, 5: 0}
    precisions: Dict[int, float] = {1: 0.0, 3: 0.0, 5: 0.0}
    diversities: Dict[int, int] = {1: 0, 3: 0, 5: 0}
    reference_hits: Dict[int, int] = {1: 0, 3: 0, 5: 0}

    for item in items:
        parsed = parser_.parse(item.query)
        query_text = item.query
        if not args.no_rewrite:
            rewrite = rewrite_query(item.query)
            query_text = rewrite.query
        docs = expert.retrieve(query_text, parsed)
        meta_list = [d.metadata or {} for d in docs]

        for k in (1, 3, 5):
            if recall_at_k(meta_list, item.expected_sources, k):
                recalls[k] += 1
            precisions[k] += _filter_precision_at_k(meta_list, parsed, k)
            diversities[k] += len({m.get("source_file") for m in meta_list[:k] if m.get("source_file")})
            if any(
                (str(m.get("section", "")).lower() == "references") or is_reference_like(docs[i].page_content or "")
                for i, m in enumerate(meta_list[:k])
            ):
                reference_hits[k] += 1

    total = max(len(items), 1)
    rows = [
        ["Recall@1", f"{recalls[1]}/{total}", f"{recalls[1] / total:.2f}"],
        ["Recall@3", f"{recalls[3]}/{total}", f"{recalls[3] / total:.2f}"],
        ["Recall@5", f"{recalls[5]}/{total}", f"{recalls[5] / total:.2f}"],
        ["FilterPrecision@1", "-", f"{precisions[1] / total:.2f}"],
        ["FilterPrecision@3", "-", f"{precisions[3] / total:.2f}"],
        ["FilterPrecision@5", "-", f"{precisions[5] / total:.2f}"],
        ["Diversity@1", "-", f"{diversities[1] / total:.2f}"],
        ["Diversity@3", "-", f"{diversities[3] / total:.2f}"],
        ["Diversity@5", "-", f"{diversities[5] / total:.2f}"],
        ["ReferencesHit@1", f"{reference_hits[1]}/{total}", f"{reference_hits[1] / total:.2f}"],
        ["ReferencesHit@3", f"{reference_hits[3]}/{total}", f"{reference_hits[3] / total:.2f}"],
        ["ReferencesHit@5", f"{reference_hits[5]}/{total}", f"{reference_hits[5] / total:.2f}"],
    ]

    headers = ["Metric", "Count", "Score"]
    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(3)]

    def format_row(row: List[str]) -> str:
        return " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(3))

    sep = "-|-".join("-" * w for w in col_widths)
    print(format_row(headers))
    print(sep)
    for row in rows:
        print(format_row(row))
    print_stats_table("Summary", [(r[0], r[2]) for r in rows])


if __name__ == "__main__":
    main()
