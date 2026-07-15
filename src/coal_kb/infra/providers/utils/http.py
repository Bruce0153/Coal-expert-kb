"""提供远程 Provider 共用的密钥解析和 HTTP 响应校验。"""

from __future__ import annotations

import os
from typing import Any


def resolve_remote_api_key(*, api_key: str | None, api_key_env: str) -> str:
    """只为远程 Provider 解析 API Key。"""
    resolved = (api_key or os.getenv(api_key_env) or "").strip()
    if not resolved:
        raise RuntimeError(f"Missing remote API key: {api_key_env}")
    return resolved


def extract_json_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    """从常见远程响应字段中提取列表。"""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []
