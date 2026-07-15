"""导出 Tokenizer Provider 正式入口。"""

from coal_kb.infra.providers.config import TokenizerProviderConfig as TokenizerConfig
from coal_kb.infra.providers.tokenizers.factory import make_tokenizer

__all__ = ["TokenizerConfig", "make_tokenizer"]
