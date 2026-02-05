from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from coal_kb.pipelines.record_pipeline import RecordPipeline
from coal_kb.settings import AppConfig


@dataclass
class KnowledgeExtractor:
    cfg: AppConfig

    def extract_records(self, docs: List[Document], *, enable_llm: bool = False, llm_provider: str = "none") -> dict:
        pipe = RecordPipeline(cfg=self.cfg, enable_llm_records=enable_llm, llm_provider=llm_provider)
        return pipe.run(docs)
