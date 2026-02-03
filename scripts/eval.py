from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from coal_kb.cli_ui import print_banner, print_stats_table
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.metadata.normalize import Ontology
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite
from coal_kb.utils.hash import stable_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    query: str
    expected_sources: List[Dict[str, Any]]


def load_eval_set(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(EvalItem(query=obj["query"], expected_sources=obj.get("expected_sources") or []))
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


def evaluate(items: List[EvalItem], retriever: ExpertRetriever, k: int) -> Dict[str, float]:
    total = max(len(items), 1)
    precision_hits = 0
    recall_hits = 0
    rr_total = 0.0
    parser = FilterParser(onto=Ontology.load("configs/schema.yaml"))

    for item in items:
        parsed = parser.parse(item.query)
        docs = retriever.retrieve(item.query, parsed)
        metas = [d.metadata or {} for d in docs[:k]]

        hit_positions = [
            idx + 1
            for idx, meta in enumerate(metas)
            if any(match_gold(gold, meta) for gold in item.expected_sources)
        ]
        if hit_positions:
            recall_hits += 1
            rr_total += 1.0 / min(hit_positions)
            precision_hits += len(hit_positions)

    precision = precision_hits / max(total * k, 1)
    recall = recall_hits / total
    mrr = rr_total / total
    return {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and log metrics.")
    parser.add_argument("--gold", default="data/eval/eval_set.jsonl")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Run id to associate with metrics.")
    parser.add_argument("--index", default=None, help="Elastic index or alias to evaluate.")
    args = parser.parse_args()

    cfg = load_config()
    print_banner("Coal KB Eval", f"backend={cfg.backend}")

    backend = cfg.backend
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
        index_name = args.index or cfg.elastic.alias_current
        vector_factory = make_elastic_retriever_factory(
            client=elastic_store.client,
            index=index_name,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            candidates=cfg.retrieval.candidates,
            rrf_k=cfg.retrieval.rrf_k,
            use_icu=cfg.elastic.enable_icu_analyzer,
        )

    if vector_factory is None:
        raise RuntimeError("No retriever factory configured.")

    k = args.k or cfg.retrieval.k
    retriever = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=k,
        rerank_enabled=cfg.retrieval.rerank_enabled,
        rerank_model=cfg.retrieval.rerank_model,
        rerank_top_n=cfg.retrieval.rerank_top_n,
        rerank_candidates=cfg.retrieval.candidates,
        rerank_device=cfg.retrieval.rerank_device,
        max_per_source=cfg.retrieval.max_per_source,
        max_relax_steps=cfg.retrieval.max_relax_steps,
        range_expand_schedule=cfg.retrieval.range_expand_schedule,
        mode=cfg.retrieval.mode,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=(backend != "elastic"),
        where_full=(backend == "elastic"),
    )

    items = load_eval_set(Path(args.gold))
    metrics = evaluate(items, retriever, k=k)

    print_stats_table(
        "Eval Summary",
        [
            ("precision@k", f"{metrics['precision_at_k']:.3f}"),
            ("recall@k", f"{metrics['recall_at_k']:.3f}"),
            ("mrr", f"{metrics['mrr']:.3f}"),
        ],
    )

    run_id = args.run_id or stable_chunk_id(str(Path(args.gold).resolve()))
    schema_hash = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))[:8]
    registry = RegistrySQLite(cfg.registry.sqlite_path)
    index_name = args.index or cfg.elastic.alias_current
    doc_count = 0
    if backend in {"elastic", "both"}:
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        doc_count = int(elastic_store.client.count(index=index_name).get("count", 0))
    registry.log_run_metrics(
        run_id=run_id,
        index_name=index_name,
        embedding_version=cfg.model_versions.embedding_version,
        schema_hash=schema_hash,
        doc_count=doc_count,
        chunks=doc_count,
        precision_at_k=metrics["precision_at_k"],
        recall_at_k=metrics["recall_at_k"],
        mrr=metrics["mrr"],
    )


if __name__ == "__main__":
    main()
