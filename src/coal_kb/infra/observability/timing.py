"""提供统一的单调时钟耗时换算。"""

from __future__ import annotations

import time

from coal_kb.infra.observability import config


def monotonic_time() -> float:
    """返回单调时钟时间点。"""
    return time.monotonic()


def elapsed_ms(started: float) -> float:
    """将当前时刻与起点之差换算为毫秒。"""
    return (time.monotonic() - started) * config.MILLISECONDS_PER_SECOND
