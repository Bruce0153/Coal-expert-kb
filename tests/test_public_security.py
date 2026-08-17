"""验证公网安全策略和匿名会话隔离。"""

from __future__ import annotations

import sqlite3

import pytest

from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore
from coal_kb.infra.security.policy import PublicSecurityPolicy


def test_public_policy_requires_strong_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COAL_KB_PUBLIC_MODE", "true")
    monkeypatch.delenv("COAL_KB_SESSION_SECRET", raising=False)
    monkeypatch.delenv("COAL_KB_ADMIN_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        PublicSecurityPolicy.from_env()


def test_public_policy_reads_limits_and_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COAL_KB_PUBLIC_MODE", "true")
    monkeypatch.setenv("COAL_KB_SESSION_SECRET", "s" * 32)
    monkeypatch.setenv("COAL_KB_ADMIN_SECRET", "a" * 24)
    monkeypatch.setenv("COAL_KB_PUBLIC_RESEARCH_ROUTES", "standard,graph")
    monkeypatch.setenv("COAL_KB_RATE_LIMIT_REQUESTS", "12")
    policy = PublicSecurityPolicy.from_env()
    assert policy.public_mode is True
    assert policy.allowed_research_routes == ("standard", "graph")
    assert policy.rate_limit_requests == 12


def test_conversations_are_isolated_by_session(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ConversationStore(str(tmp_path / "conversation.db"))
    alice = ConversationService(store).for_session("alice")
    bob = ConversationService(store).for_session("bob")

    conversation = alice.create_conversation(title="private")
    alice.add_message(conversation_id=conversation.conversation_id, role="user", content="secret")

    assert [item.conversation_id for item in alice.list_conversations()] == [conversation.conversation_id]
    assert bob.list_conversations() == []
    with pytest.raises(KeyError):
        bob.list_messages(conversation.conversation_id)
    assert bob.delete_conversation(conversation.conversation_id) is False
    assert len(alice.list_messages(conversation.conversation_id)) == 1


def test_existing_conversation_database_is_migrated_to_legacy_session(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE conversations ("
        "conversation_id VARCHAR(64) PRIMARY KEY, title TEXT, created_at DATETIME, updated_at DATETIME)"
    )
    connection.execute(
        "CREATE TABLE conversation_messages ("
        "message_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64), role VARCHAR(16), "
        "content TEXT, metadata_json TEXT, created_at DATETIME)"
    )
    connection.execute(
        "INSERT INTO conversations(conversation_id, title, created_at, updated_at) "
        "VALUES ('old', 'Old conversation', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )
    connection.commit()
    connection.close()

    store = ConversationStore(str(path))
    assert store.get_conversation("old") is not None
    assert store.get_conversation("old", session_id="new-browser") is None
