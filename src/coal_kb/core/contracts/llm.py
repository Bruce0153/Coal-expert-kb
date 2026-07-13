from __future__ import annotations

from typing import Any, Protocol


class ChatModelProvider(Protocol):
    """Minimal chat-model capability required by generation and query rewriting."""

    def invoke(self, input: Any, **kwargs: Any) -> Any: ...
