from datetime import datetime

from coal_kb.chat.memory import prepare_history_context
from coal_kb.conversation.models import ConversationMessage


def _msg(role: str, content: str) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"{role}-1",
        conversation_id="conv-1",
        role=role,  # type: ignore[arg-type]
        content=content,
        metadata={},
        created_at=datetime.utcnow(),
    )


def test_prepare_history_context_uses_history_for_follow_up():
    messages = [
        _msg("user", "How does steam gasification affect NH3 and HCN at 1200 K?"),
        _msg("assistant", "Steam gasification often keeps both NH3 and HCN relevant under high-temperature conditions."),
    ]
    prepared = prepare_history_context(messages, "What about CO2 instead?")
    assert prepared.used_history is True
    assert "Previous user focus" in prepared.retrieval_query
    assert prepared.reason == "follow_up_rewrite"


def test_prepare_history_context_leaves_standalone_query_alone():
    prepared = prepare_history_context([], "steam gasification NH3 1200K")
    assert prepared.used_history is False
    assert prepared.retrieval_query == "steam gasification NH3 1200K"
