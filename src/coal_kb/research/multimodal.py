"""统一组织文本、表格和图像类证据并执行可解释重排。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document

from coal_kb.research.models import RouteResult
from coal_kb.utils.documents import document_key

BaseRoute = Callable[[], RouteResult]


@dataclass
class MultimodalRoute:
    """根据问题和元数据标记证据模态，不调用未配置的视觉模型。"""

    requested_boost: float = 1.5
    secondary_boost: float = 0.15

    def process(self, query: str, base_route: BaseRoute) -> RouteResult:
        base = base_route()
        requested = self.requested_modalities(query)
        scored: list[tuple[Document, str, float, int]] = []
        distribution: dict[str, int] = {}
        for rank, document in enumerate(base.documents):
            modality = self.infer_modality(document)
            distribution[modality] = distribution.get(modality, 0) + 1
            score = 1.0 / (rank + 1)
            if modality in requested:
                score += self.requested_boost
            elif modality != "text":
                score += self.secondary_boost
            metadata = dict(document.metadata or {})
            metadata["research_modality"] = modality
            metadata["multimodal_score"] = round(score, 6)
            scored.append(
                (
                    Document(page_content=document.page_content, metadata=metadata),
                    modality,
                    score,
                    rank,
                )
            )
        scored.sort(key=lambda item: (-item[2], item[3], document_key(item[0])))
        documents = [item[0] for item in scored]
        trace = {
            "route": "multimodal",
            "base": base.trace,
            "multimodal": {
                "requested_modalities": sorted(requested),
                "available_modalities": distribution,
                "selected_chunk_ids": [document_key(item) for item in documents],
                "selected_modalities": [item[1] for item in scored],
                "scores": {
                    document_key(item[0]): round(item[2], 6)
                    for item in scored
                },
            },
        }
        return RouteResult(documents=documents, trace=trace)

    @staticmethod
    def requested_modalities(query: str) -> set[str]:
        lowered = query.lower()
        requested: set[str] = set()
        if re.search(r"图|图像|曲线|示意图|figure|image|chart|diagram|plot", lowered):
            requested.add("image")
        if re.search(r"表|表格|table|row|column|cell", lowered):
            requested.add("table")
        if not requested:
            requested.add("text")
        return requested

    @staticmethod
    def infer_modality(document: Document) -> str:
        metadata = document.metadata or {}
        explicit = str(metadata.get("modality") or metadata.get("research_modality") or "").lower()
        if explicit in {"text", "table", "image"}:
            return explicit
        section = str(metadata.get("section") or metadata.get("heading_path") or "").lower()
        if metadata.get("table_id") or "table" in section or "表格" in section:
            return "table"
        if (
            metadata.get("figure_id")
            or metadata.get("image_path")
            or any(marker in section for marker in ("figure", "image", "图像", "插图"))
        ):
            return "image"
        text = (document.page_content or "").lstrip().lower()
        if re.match(r"^(figure|fig\.|图\s*\d+|chart)", text):
            return "image"
        return "text"
