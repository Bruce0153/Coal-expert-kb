from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore


def test_conversation_creation_and_message_persistence(tmp_path):
    db_path = tmp_path / "chat.db"
    service = ConversationService(ConversationStore(str(db_path)))

    conversation = service.create_conversation(title="Steam NH3 discussion")
    user_message = service.add_message(
        conversation_id=conversation.conversation_id,
        role="user",
        content="How does steam gasification affect NH3?",
    )
    assistant_message = service.add_message(
        conversation_id=conversation.conversation_id,
        role="assistant",
        content="It depends on temperature.",
        metadata={"confidence_score": 0.5},
    )

    listed = service.list_conversations()
    messages = service.list_messages(conversation.conversation_id)

    assert listed[0].conversation_id == conversation.conversation_id
    assert listed[0].message_count == 2
    assert messages[0].message_id == user_message.message_id
    assert messages[1].metadata["confidence_score"] == 0.5
    assert assistant_message.role == "assistant"
