"""聊天历史兼容入口，编排器按需加载以避免循环导入。"""

from .memory import PreparedHistory, prepare_history_context

__all__ = ["PreparedHistory", "prepare_history_context", "ChatOrchestrator", "ChatTurnResult"]


def __getattr__(name: str):
    if name in {"ChatOrchestrator", "ChatTurnResult"}:
        from .orchestrator import ChatOrchestrator, ChatTurnResult

        return {"ChatOrchestrator": ChatOrchestrator, "ChatTurnResult": ChatTurnResult}[name]
    raise AttributeError(name)
