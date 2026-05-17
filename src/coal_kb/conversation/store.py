from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, Text, create_engine, delete, desc, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .models import ConversationMessage, ConversationSummary


class Base(DeclarativeBase):
    pass


class ConversationModel(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MessageModel(Base):
    __tablename__ = "conversation_messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16), index=True)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ConversationStore:
    def __init__(self, sqlite_path: str) -> None:
        self._engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
        Base.metadata.create_all(self._engine)

    def create_conversation(self, *, title: str | None = None) -> ConversationSummary:
        now = datetime.utcnow()
        conversation_id = uuid4().hex
        conversation_title = (title or "New conversation").strip() or "New conversation"
        with Session(self._engine) as session:
            model = ConversationModel(
                conversation_id=conversation_id,
                title=conversation_title,
                created_at=now,
                updated_at=now,
            )
            session.add(model)
            session.commit()
        return ConversationSummary(
            conversation_id=conversation_id,
            title=conversation_title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> ConversationSummary | None:
        with Session(self._engine) as session:
            model = session.get(ConversationModel, conversation_id)
            if model is None:
                return None
            count_stmt = select(func.count(MessageModel.message_id)).where(MessageModel.conversation_id == conversation_id)
            count = int(session.execute(count_stmt).scalar_one())
            return ConversationSummary(
                conversation_id=model.conversation_id,
                title=model.title,
                created_at=model.created_at,
                updated_at=model.updated_at,
                message_count=count,
            )

    def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        with Session(self._engine) as session:
            stmt = select(ConversationModel).order_by(desc(ConversationModel.updated_at)).limit(limit)
            models = session.execute(stmt).scalars().all()
            counts_stmt = (
                select(MessageModel.conversation_id, func.count(MessageModel.message_id))
                .group_by(MessageModel.conversation_id)
            )
            counts = {row[0]: int(row[1]) for row in session.execute(counts_stmt).all()}
            return [
                ConversationSummary(
                    conversation_id=model.conversation_id,
                    title=model.title,
                    created_at=model.created_at,
                    updated_at=model.updated_at,
                    message_count=counts.get(model.conversation_id, 0),
                )
                for model in models
            ]

    def delete_conversation(self, conversation_id: str) -> bool:
        with Session(self._engine) as session:
            exists = session.get(ConversationModel, conversation_id)
            if exists is None:
                return False
            session.execute(delete(MessageModel).where(MessageModel.conversation_id == conversation_id))
            session.execute(delete(ConversationModel).where(ConversationModel.conversation_id == conversation_id))
            session.commit()
            return True

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ConversationMessage:
        now = datetime.utcnow()
        message_id = uuid4().hex
        with Session(self._engine) as session:
            conversation = session.get(ConversationModel, conversation_id)
            if conversation is None:
                raise KeyError(f"Conversation not found: {conversation_id}")
            model = MessageModel(
                message_id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
                created_at=now,
            )
            session.add(model)
            conversation.updated_at = now
            session.commit()
        return ConversationMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            metadata=metadata or {},
            created_at=now,
        )

    def list_messages(self, conversation_id: str, *, limit: int | None = None) -> list[ConversationMessage]:
        with Session(self._engine) as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.conversation_id == conversation_id)
                .order_by(MessageModel.created_at.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            models = session.execute(stmt).scalars().all()
            return [
                ConversationMessage(
                    message_id=model.message_id,
                    conversation_id=model.conversation_id,
                    role=model.role,  # type: ignore[arg-type]
                    content=model.content,
                    metadata=json.loads(model.metadata_json) if model.metadata_json else {},
                    created_at=model.created_at,
                )
                for model in models
            ]

    def update_title(self, conversation_id: str, title: str) -> None:
        clean_title = title.strip() or "New conversation"
        with Session(self._engine) as session:
            model = session.get(ConversationModel, conversation_id)
            if model is None:
                raise KeyError(f"Conversation not found: {conversation_id}")
            model.title = clean_title
            model.updated_at = datetime.utcnow()
            session.commit()
