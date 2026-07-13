"""提供 FastAPI 应用工厂，避免导入数据模型时启动完整运行时。"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
