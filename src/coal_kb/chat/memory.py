from __future__ import annotations

import re
from dataclasses import dataclass

from coal_kb.conversation.models import ConversationMessage

_FOLLOW_UP_PATTERNS = [
    r"^(and|what about|how about|then|also|compare|now|those|them|it|their)\b",
    r"^(那|那么|那在|那对于|继续|再说|还有|对比|比较|这些|它们|这个)\b",
]


@dataclass
class PreparedHistory:
    retrieval_query: str
    answer_history: str
    used_history: bool
    reason: str


def _is_follow_up(query: str) -> bool:
    normalized = query.strip().lower()
    if len(normalized.split()) <= 8:
        for pattern in _FOLLOW_UP_PATTERNS:
            if re.search(pattern, normalized):
                return True
    return any(token in normalized for token in ["what about", "how about", "compare", "那", "继续", "对比"])


def _assistant_summary(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= 220:
        return collapsed
    return collapsed[:217].rstrip() + "..."


def prepare_history_context(messages: list[ConversationMessage], query: str, *, max_turns: int = 4) -> PreparedHistory:
    recent_messages = messages[-max_turns:] if messages else []
    follow_up = _is_follow_up(query) and bool(recent_messages)
    answer_history_lines: list[str] = []
    recent_user_messages = [msg.content for msg in recent_messages if msg.role == "user"][-2:]
    recent_assistant = next((msg.content for msg in reversed(recent_messages) if msg.role == "assistant"), "")

    for message in recent_messages:
        role = message.role.upper()
        answer_history_lines.append(f"{role}: {' '.join(message.content.split())[:240]}")

    answer_history = "\n".join(answer_history_lines)

    if not follow_up:
        return PreparedHistory(
            retrieval_query=query,
            answer_history=answer_history,
            used_history=False,
            reason="standalone_query",
        )

    parts = [query]
    if recent_user_messages:
        parts.append("Previous user focus: " + " | ".join(recent_user_messages))
    if recent_assistant:
        parts.append("Previous assistant summary: " + _assistant_summary(recent_assistant))

    return PreparedHistory(
        retrieval_query="\n".join(parts),
        answer_history=answer_history,
        used_history=True,
        reason="follow_up_rewrite",
    )
