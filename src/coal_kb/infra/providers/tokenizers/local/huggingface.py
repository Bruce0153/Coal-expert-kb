"""使用本地 HuggingFace Tokenizer 统计 Token。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coal_kb.infra.providers.config import LocalProviderConfig


@dataclass
class LocalTokenizer:
    """延迟加载并复用本地 Tokenizer。"""

    config: LocalProviderConfig
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model,
                local_files_only=True,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """返回本地模型真实 Token 数。"""
        if not text:
            return 0
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))
