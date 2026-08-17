"""使用轻量 tiktoken 编码器统计生产环境 Token。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coal_kb.infra.providers.config import LocalProviderConfig


@dataclass
class TiktokenTokenizer:
    """不加载模型权重的本地 Token 计数器。"""

    config: LocalProviderConfig
    _encoding: Any = field(default=None, init=False, repr=False)

    def _get_encoding(self):
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self.config.model or "cl100k_base")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._get_encoding().encode(text, disallowed_special=()))
