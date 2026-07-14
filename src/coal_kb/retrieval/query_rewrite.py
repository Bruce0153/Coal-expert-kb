from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from coal_kb.infra.providers.llm import LLMConfig, make_chat_llm

logger = logging.getLogger(__name__)


_MECH_KEYWORDS = [
    "formation mechanism",
    "mechanism",
    "生成机理",
    "机理",
    "形成机理",
]

_MECH_EXPANSIONS = [
    "formation mechanism",
    "reaction pathway",
    "reaction network",
    "precursor conversion",
    "intermediate species",
]

# 只保留“显式 stage 词”的扩展，不再用简单子串误伤“氮氧化物”
_STAGE_PATTERNS = {
    "热解": [r"热解", r"\bpyrolysis\b"],
    "气化": [r"气化", r"\bgasification\b"],
    "氧化": [r"(^|[\s,，。;；()（）])氧化($|[\s,，。;；()（）])", r"\boxidation\b", r"\boxidative\b"],
    "燃烧": [r"燃烧", r"\bcombustion\b", r"\bburning\b"],
    "点火": [r"点火", r"\bignition\b"],
}

_STAGE_EXPANSIONS = {
    "热解": ["pyrolysis"],
    "气化": ["gasification"],
    "氧化": ["oxidation"],
    "燃烧": ["combustion"],
    "点火": ["ignition"],
}

_MECH_PATTERNS = {
    "生成机理": [r"生成机理", r"形成机理", r"\bformation mechanism\b"],
    "机理": [r"(^|[\s,，。;；()（）])机理($|[\s,，。;；()（）])", r"\bmechanism\b"],
}

_MECH_RULE_EXPANSIONS = {
    "生成机理": ["formation mechanism", "reaction pathway"],
    "机理": ["mechanism", "reaction pathway"],
}


@dataclass
class QueryRewriteResult:
    query: str
    reason: str = ""


def _has_any_pattern(text: str, patterns: list[str], *, ignore_case: bool = True) -> bool:
    flags = re.I if ignore_case else 0
    return any(re.search(p, text, flags=flags) for p in patterns)


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
    expansions: list[str] = []

    # 显式 stage 词才扩展
    for zh_key, patterns in _STAGE_PATTERNS.items():
        if _has_any_pattern(base, patterns):
            expansions.extend(_STAGE_EXPANSIONS[zh_key])

    # 机理类扩展
    for zh_key, patterns in _MECH_PATTERNS.items():
        if _has_any_pattern(base, patterns):
            expansions.extend(_MECH_RULE_EXPANSIONS[zh_key])

    if expansions:
        expanded = base + " " + " ".join(sorted(set(expansions)))
        return QueryRewriteResult(query=expanded, reason="zh_rules")

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