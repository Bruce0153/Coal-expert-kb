"""兼容旧 ChatOrchestrator 导入和 monkeypatch 路径。"""

from __future__ import annotations

from typing import Optional

from coal_kb.application import chat as _chat

execute_query = _chat.execute_query
log_query = _chat.log_query

ChatTurnResult = _chat.ChatTurnResult


class ChatOrchestrator(_chat.ChatOrchestrator):
    """在调用前同步旧模块注入的函数，保持测试和扩展兼容。"""

    def chat(
        self,
        *,
        query: str,
        conversation_id: Optional[str] = None,
        enable_llm: bool = False,
        save_trace: bool = False,
        debug: bool = False,
    ) -> ChatTurnResult:
        _chat.execute_query = execute_query
        _chat.log_query = log_query
        return super().chat(
            query=query,
            conversation_id=conversation_id,
            enable_llm=enable_llm,
            save_trace=save_trace,
            debug=debug,
        )


__all__ = ["ChatOrchestrator", "ChatTurnResult", "execute_query", "log_query"]
