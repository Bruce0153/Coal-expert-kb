"""通过远程 Tokenizer API 统计 Token。"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key


@dataclass
class RemoteTokenizer:
    """保存远程 Tokenizer API 配置。"""

    config: RemoteProviderConfig

    def count_tokens(self, text: str) -> int:
        """调用远程 tokenize 接口返回精确 Token 数。"""
        if not text:
            return 0
        endpoint = self.config.endpoint or "/tokenize"
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            json={"model": self.config.model, "text": text},
            headers={"Authorization": f"Bearer {resolve_remote_api_key(api_key=self.config.api_key, api_key_env=self.config.api_key_env)}"},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload.get("count"), int):
            return payload["count"]
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("Remote tokenizer response must contain count or tokens")
