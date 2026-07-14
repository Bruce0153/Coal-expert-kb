"""评估两阶段检索的召回、约束精度、多样性与参考文献污染。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.infra.providers.rerank import make_reranker
from coal_kb.ingestion.chunking.sectioner import is_reference_like
from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.interfaces.cli.ui import print_banner, print_stats_table
from coal_kb.retrieval.query.filter_parser import FilterParser
from coal_kb.retrieval.query.rewrite import rewrite_query
from coal_kb.retrieval.service import ExpertRetriever


@dataclass
class EvalItem:
    query: str
    expected_sources: list[dict[str, Any]]
    expected_stage: str | None = None


@dataclass
class EvalRetrieval:
    cfg: AppConfig
    gold_path: Path
    k: int
    disable_rewrite: bool

    @staticmethod
    def _load_eval_set(path: Path, *, desc: str) -> list[EvalItem]:
        lines = path.read_text(encoding="utf-8").splitlines()
        items: list[EvalItem] = []
        for line in tqdm(lines, total=len(lines), desc=desc):
            if not line.strip():
                continue
            payload: dict[str, Any] = json.loads(line)
            items.append(
                EvalItem(
                    query=str(payload["query"]),
                    expected_sources=list(payload.get("expected_sources") or []),
                    expected_stage=payload.get("expected_stage"),
                )
            )
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

    @staticmethod
    def _range_overlap(
        metadata: dict[str, Any],
        query_range: list[float] | None,
        *,
        key_point: str,
        key_min: str,
        key_max: str,
    ) -> bool:
        if query_range is None:
            return True
        lower, upper = float(query_range[0]), float(query_range[1])
        document_min = metadata.get(key_min)
        document_max = metadata.get(key_max)
        if document_min is not None and document_max is not None:
            return max(float(document_min), lower) <= min(float(document_max), upper)
        point = metadata.get(key_point)
        if point is None:
            return False
        return lower <= float(point) <= upper

    @classmethod
    def _filter_match(cls, metadata: dict[str, Any], parsed: dict[str, Any]) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        stage = parsed.get("stage")
        if stage and stage != "unknown":
            checks["stage"] = str(metadata.get("stage")) == stage
        gas_agents = parsed.get("gas_agent") or []
        if gas_agents:
            checks["gas_agent"] = any(metadata.get(f"gas_{agent}") for agent in gas_agents)
        targets = parsed.get("targets") or []
        if targets:
            checks["targets"] = any(metadata.get(f"has_{target}") for target in targets)
        temperature_range = parsed.get("T_range_K")
        if temperature_range:
            checks["T_range_K"] = cls._range_overlap(
                metadata,
                temperature_range,
                key_point="T_K",
                key_min="T_min_K",
                key_max="T_max_K",
            )
        pressure_range = parsed.get("P_range_MPa")
        if pressure_range:
            checks["P_range_MPa"] = cls._range_overlap(
                metadata,
                pressure_range,
                key_point="P_MPa",
                key_min="P_min_MPa",
                key_max="P_max_MPa",
            )
        return checks

    @classmethod
    def _recall_at_k(
        cls,
        documents: list[dict[str, Any]],
        expected_sources: list[dict[str, Any]],
        k: int,
    ) -> bool:
        return any(
            cls._match_gold(gold, metadata)
            for metadata in documents[:k]
            for gold in expected_sources
        )

    @classmethod
    def _filter_precision_at_k(
        cls,
        documents: list[dict[str, Any]],
        parsed: dict[str, Any],
        k: int,
    ) -> float:
        if not documents:
            return 0.0
        hits = 0
        for metadata in documents[:k]:
            checks = cls._filter_match(metadata, parsed)
            if checks and all(checks.values()):
                hits += 1
        return hits / min(k, len(documents))

    @staticmethod
    def _format_table(rows: list[list[str]]) -> str:
        headers = ["Metric", "Count", "Score"]
        widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(3)]
        lines = [
            " | ".join(headers[index].ljust(widths[index]) for index in range(3)),
            "-|-".join("-" * width for width in widths),
        ]
        lines.extend(" | ".join(str(row[index]).ljust(widths[index]) for index in range(3)) for row in rows)
        return "\n".join(lines)

    def _build_retriever(self) -> ExpertRetriever:
        if self.cfg.backend != "elastic":
            raise ValueError("Two-stage retrieval evaluation requires backend=elastic")
        elastic_store = ElasticStore(
            host=self.cfg.elastic.host,
            verify_certs=self.cfg.elastic.verify_certs,
            timeout_s=self.cfg.elastic.timeout_s,
        )
        vector_factory = elastic_store.make_retriever_factory(
            index=self.cfg.elastic.alias_current,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            candidates=self.k,
            rrf_k=self.cfg.retrieval.rrf_k,
            use_icu=self.cfg.elastic.enable_icu_analyzer,
        )
        rerank_enabled = bool(self.cfg.retrieval.rerank_enabled)
        reranker = make_reranker(self.cfg) if rerank_enabled else None
        return ExpertRetriever(
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
            use_fuse=False,
            where_full=True,
            two_stage_enabled=self.cfg.retrieval.two_stage.enabled,
            parent_k_candidates=self.cfg.retrieval.two_stage.parent_k_candidates,
            parent_k_final=self.cfg.retrieval.two_stage.parent_k_final,
            max_parents=self.cfg.retrieval.two_stage.max_parents,
            child_k_candidates=self.cfg.retrieval.two_stage.child_k_candidates,
            child_k_final=self.cfg.retrieval.two_stage.child_k_final,
            allow_relax_in_stage2=self.cfg.retrieval.two_stage.allow_relax_in_stage2,
            elastic_store=elastic_store,
            elastic_index=self.cfg.elastic.alias_current,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            elastic_use_icu=self.cfg.elastic.enable_icu_analyzer,
            tenant_id=self.cfg.tenancy.default_tenant_id if self.cfg.tenancy.enabled else None,
        )

    def process(self) -> list[list[str]]:
        print_banner("Coal KB Retrieval Eval", f"backend={self.cfg.backend}")
        parser = FilterParser(onto=Ontology.load("configs/schema.yaml"))
        retriever = self._build_retriever()
        items = self._load_eval_set(self.gold_path, desc=self.__class__.__name__)
        recalls = {1: 0, 3: 0, 5: 0}
        precisions = {1: 0.0, 3: 0.0, 5: 0.0}
        diversities = {1: 0, 3: 0, 5: 0}
        reference_hits = {1: 0, 3: 0, 5: 0}
        parent_recall = 0
        child_recall = 0
        for item in tqdm(items, total=len(items), desc=self.__class__.__name__):
            parsed = parser.parse(item.query)
            query_text = item.query if self.disable_rewrite else rewrite_query(item.query).query
            trace: dict[str, Any] = {}
            documents = retriever.retrieve(query_text, parsed, trace=trace)
            if set(trace.get("stage1_parent_ids", [])):
                parent_recall += 1
            metadata_rows = [document.metadata or {} for document in documents]
            for current_k in (1, 3, 5):
                if self._recall_at_k(metadata_rows, item.expected_sources, current_k):
                    recalls[current_k] += 1
                    if current_k == 5:
                        child_recall += 1
                precisions[current_k] += self._filter_precision_at_k(
                    metadata_rows,
                    parsed.compat_where,
                    current_k,
                )
                diversities[current_k] += len(
                    {
                        metadata.get("source_file")
                        for metadata in metadata_rows[:current_k]
                        if metadata.get("source_file")
                    }
                )
                if any(
                    str(metadata.get("section", "")).lower() == "references"
                    or is_reference_like(documents[index].page_content or "")
                    for index, metadata in enumerate(metadata_rows[:current_k])
                ):
                    reference_hits[current_k] += 1
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
            ["ParentsRecall(any)", f"{parent_recall}/{total}", f"{parent_recall / total:.2f}"],
            ["ChildrenRecall@5", f"{child_recall}/{total}", f"{child_recall / total:.2f}"],
        ]
        print(self._format_table(rows))
        print_stats_table("Summary", [(row[0], row[2]) for row in rows])
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality metrics.")
    parser.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()
    EvalRetrieval(
        cfg=load_config(),
        gold_path=Path(args.gold),
        k=args.k,
        disable_rewrite=args.no_rewrite,
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/eval_retrieval.py
