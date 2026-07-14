"""从向量库检索文档块并抽取结构化实验记录。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.records.pipeline import RecordPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExtractRecords:
    cfg: AppConfig
    limit: int
    enable_llm: bool
    llm_provider: str

    def process(self) -> dict:
        store = ChromaStore(
            persist_dir=self.cfg.paths.chroma_dir,
            collection_name=self.cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            embedding_model=self.cfg.embeddings.model,
        )
        retriever = store.as_retriever(k=self.limit, where=None)
        # 当前保持原检索策略，后续由数据库扫描接口替代。
        query = "gasification pyrolysis experimental conditions table results"
        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(query)
        else:
            documents = retriever.get_relevant_documents(query)
        provider = self.llm_provider
        if self.enable_llm and provider == "none":
            provider = self.cfg.llm.provider
        pipeline = RecordPipeline(
            cfg=self.cfg,
            enable_llm_records=self.enable_llm,
            llm_provider=provider,
        )
        return pipeline.run(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ExperimentRecords into SQLite.")
    parser.add_argument("--limit", type=int, default=200, help="Max chunks to sample for extraction.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM record extraction.")
    parser.add_argument(
        "--llm-provider",
        default="none",
        choices=["none", "openai", "openai_compatible", "dashscope"],
    )
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    step = ExtractRecords(
        cfg=cfg,
        limit=args.limit,
        enable_llm=args.llm,
        llm_provider=args.llm_provider,
    )
    print(step.process())


if __name__ == "__main__":
    main()

# 运行命令：python scripts/extract_records.py
