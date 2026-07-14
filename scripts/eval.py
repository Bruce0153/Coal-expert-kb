"""评估检索 Precision、Recall 与 MRR 并记录运行指标。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.persistence.registry import RegistrySQLite
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.infra.providers.rerank import make_reranker
from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.interfaces.cli.ui import print_banner, print_stats_table
from coal_kb.retrieval.query.filter_parser import FilterParser
from coal_kb.retrieval.service import ExpertRetriever
from coal_kb.utils.hash import stable_chunk_id


@dataclass
class EvalItem:
    query: str
    expected_sources: list[dict[str, Any]]


@dataclass
class Eval:
    cfg: AppConfig
    gold_path: Path
    k: int
    run_id: str | None
    index_name: str | None

    @staticmethod
    def _load_eval_set(path: Path, *, desc: str) -> list[EvalItem]:
        lines = path.read_text(encoding="utf-8").splitlines()
        items: list[EvalItem] = []
        for line in tqdm(lines, total=len(lines), desc=desc):
            if not line.strip():
                continue
            payload: dict[str, Any] = json.loads(line)
            items.append(EvalItem(query=str(payload["query"]), expected_sources=list(payload.get("expected_sources") or [])))
        return items

    @staticmethod
    def _match_gold(gold: dict[str, Any], metadata: dict[str, Any]) -> bool:
        chunk_id = gold.get("chunk_id")
        if chunk_id and str(metadata.get("chunk_id")) == str(chunk_id):
            return True
        source = str(metadata.get("source_file", "")).lower()
        expected_source = str(gold.get("source_file", "")).lower()
        if expected_source and expected_source not in source:
            return False
        page = gold.get("page")
        if page is not None:
            return metadata.get("page") == page
        return bool(expected_source)

    def _build_retriever(self) -> tuple[ExpertRetriever, ElasticStore | None, str]:
        backend = self.cfg.backend
        if backend not in {"elastic", "chroma", "both"}:
            raise ValueError(f"Unsupported backend: {backend}")
        vector_factory = None
        elastic_store: ElasticStore | None = None
        resolved_index = self.index_name or self.cfg.elastic.alias_current
        if backend in {"chroma", "both"}:
            store = ChromaStore(
                persist_dir=self.cfg.paths.chroma_dir,
                collection_name=self.cfg.chroma.collection_name,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                embedding_model=self.cfg.embedding.model_name,
            )
            vector_factory = store.as_retriever
        if backend in {"elastic", "both"}:
            elastic_store = ElasticStore(
                host=self.cfg.elastic.host,
                verify_certs=self.cfg.elastic.verify_certs,
                timeout_s=self.cfg.elastic.timeout_s,
            )
            vector_factory = elastic_store.make_retriever_factory(
                index=resolved_index,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                candidates=self.k,
                rrf_k=self.cfg.retrieval.rrf_k,
                use_icu=self.cfg.elastic.enable_icu_analyzer,
            )
        if vector_factory is None:
            raise RuntimeError("No retriever factory configured.")
        rerank_enabled = bool(self.cfg.retrieval.rerank_enabled)
        reranker = make_reranker(self.cfg) if rerank_enabled else None
        retriever = ExpertRetriever(
            vector_retriever_factory=vector_factory,
            k=self.k,
            rerank_enabled=rerank_enabled,
            rerank_top_n=self.cfg.retrieval.rerank_top_n,
            reranker=reranker,
            max_per_source=self.cfg.retrieval.max_per_source,
            max_relax_steps=self.cfg.retrieval.max_relax_steps,
            range_expand_schedule=self.cfg.retrieval.range_expand_schedule,
            mode=self.cfg.retrieval.mode,
            drop_sections=self.cfg.retrieval.drop_sections,
            drop_reference_like=self.cfg.retrieval.drop_reference_like,
            use_fuse=(backend != "elastic"),
            where_full=(backend == "elastic"),
        )
        return retriever, elastic_store, resolved_index

    def _evaluate(self, items: list[EvalItem], retriever: ExpertRetriever) -> dict[str, float]:
        total = max(len(items), 1)
        precision_hits = 0
        recall_hits = 0
        reciprocal_rank_total = 0.0
        parser = FilterParser(onto=Ontology.load("configs/schema.yaml"))
        for item in tqdm(items, total=len(items), desc=self.__class__.__name__):
            parsed = parser.parse(item.query)
            documents = retriever.retrieve(item.query, parsed)
            metadata_rows = [document.metadata or {} for document in documents[: self.k]]
            positions = [
                index + 1
                for index, metadata in enumerate(metadata_rows)
                if any(self._match_gold(gold, metadata) for gold in item.expected_sources)
            ]
            if positions:
                recall_hits += 1
                reciprocal_rank_total += 1.0 / min(positions)
                precision_hits += len(positions)
        return {
            "precision_at_k": precision_hits / max(total * self.k, 1),
            "recall_at_k": recall_hits / total,
            "mrr": reciprocal_rank_total / total,
        }

    def process(self) -> dict[str, float]:
        print_banner("Coal KB Eval", f"backend={self.cfg.backend}")
        retriever, elastic_store, resolved_index = self._build_retriever()
        items = self._load_eval_set(self.gold_path, desc=self.__class__.__name__)
        metrics = self._evaluate(items, retriever)
        print_stats_table(
            "Eval Summary",
            [
                ("precision@k", f"{metrics['precision_at_k']:.3f}"),
                ("recall@k", f"{metrics['recall_at_k']:.3f}"),
                ("mrr", f"{metrics['mrr']:.3f}"),
            ],
        )
        run_id = self.run_id or stable_chunk_id(str(self.gold_path.resolve()))
        schema_hash = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))[:8]
        document_count = 0
        if elastic_store is not None and self.cfg.backend in {"elastic", "both"}:
            document_count = int(elastic_store.client.count(index=resolved_index).get("count", 0))
        RegistrySQLite(self.cfg.registry.sqlite_path).log_run_metrics(
            run_id=run_id,
            index_name=resolved_index,
            embedding_version=self.cfg.model_versions.embedding_version,
            schema_hash=schema_hash,
            doc_count=document_count,
            chunks=document_count,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            mrr=metrics["mrr"],
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and log metrics.")
    parser.add_argument("--gold", default="data/eval/eval_set.jsonl")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Run id to associate with metrics.")
    parser.add_argument("--index", default=None, help="Elastic index or alias to evaluate.")
    args = parser.parse_args()
    cfg = load_config()
    Eval(
        cfg=cfg,
        gold_path=Path(args.gold),
        k=int(args.k or cfg.retrieval.k),
        run_id=args.run_id,
        index_name=args.index,
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/eval.py
