from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from coal_kb.llm.factory import LLMConfig, make_chat_llm

logger = logging.getLogger(__name__)


_MECH_KEYWORDS = [
    "formation mechanism",
    "mechanism",
    "生成机理",
    "机理",
    "形成机理",
]

_MECH_EXPANSIONS = [
    "phenols formation mechanism",
    "phenolic compounds",
    "tar phenolic compounds",
    "cleavage",
    "condensation",
    "secondary reactions",
    "oxygen-containing functional groups",
    "ether bond cleavage",
]


@dataclass
class QueryRewriteResult:
    query: str
    reason: str = ""


def rewrite_query(
    query: str,
    *,
    enable_llm: bool = False,
    llm_config: Optional[LLMConfig] = None,
) -> QueryRewriteResult:
    base = query.strip()
    if not base:
        return QueryRewriteResult(query=query)

    lower = base.lower()
    if any(k in lower for k in _MECH_KEYWORDS):
        expanded = base + " " + " ".join(_MECH_EXPANSIONS)
        return QueryRewriteResult(query=expanded, reason="mechanism_rules")

    if not enable_llm or llm_config is None:
        return QueryRewriteResult(query=base)

    try:
        model = make_chat_llm(llm_config)
    except Exception as e:
        logger.warning("Query rewrite LLM unavailable: %s", e)
        return QueryRewriteResult(query=base)

    prompt = (
        "You are expanding a search query for scientific literature retrieval. "
        "Return a JSON object with a single key 'query' that adds concise academic synonyms. "
        "Do not add new constraints or change intent."
    )
    user = f"Query: {base}"
    try:
        rsp = model.invoke(prompt + "\n" + user)
        content = getattr(rsp, "content", None) or ""
        obj = json.loads(content)
        rewritten = str(obj.get("query", "")).strip()
        if rewritten:
            return QueryRewriteResult(query=rewritten, reason="llm")
    except Exception as e:
        logger.warning("Query rewrite failed: %s", e)

    return QueryRewriteResult(query=base)
