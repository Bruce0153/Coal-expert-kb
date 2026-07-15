"""通过可配置 HTTP API 执行远程重排序。"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import extract_json_list, resolve_remote_api_key


@dataclass
class RemoteReranker:
    """封装远程重排序连接状态。"""

    config: RemoteProviderConfig

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """调用远程重排序 API 并返回正式排序结果。"""
        if not docs:
            return []
        endpoint = self.config.endpoint or "/rerank"
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": [doc.page_content or "" for doc in docs],
            "top_n": min(top_k, len(docs)),
        }
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {resolve_remote_api_key(api_key=self.config.api_key, api_key_env=self.config.api_key_env)}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = extract_json_list(data, "results", "data")
        indexed_scores: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            score = item.get("relevance_score", item.get("score", 0.0))
            indexed_scores.append((item["index"], float(score)))
        if not indexed_scores:
            scores = extract_json_list(data, "scores")
            indexed_scores = [(index, float(score)) for index, score in enumerate(scores)]
        if not indexed_scores:
            raise ValueError("Remote rerank response does not contain ranked indices or scores")
        ranked = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
        return [docs[index] for index, _ in ranked if 0 <= index < len(docs)][:top_k]
