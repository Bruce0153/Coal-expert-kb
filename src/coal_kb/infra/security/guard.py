"""单实例公网请求限流与并发保护。"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterator
from threading import BoundedSemaphore, Lock

from fastapi import HTTPException, Request, status

from .policy import PublicSecurityPolicy


class PublicRequestGuard:
    """面向单实例部署的轻量保护；多实例部署时应替换为共享限流存储。"""

    def __init__(self, policy: PublicSecurityPolicy) -> None:
        self.policy = policy
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._semaphore = BoundedSemaphore(policy.max_concurrent_queries)

    def protect(self, request: Request) -> Iterator[None]:
        if not self.policy.public_mode:
            yield
            return

        key = getattr(request.state, "session_id", None)
        if not key and request.client:
            key = request.client.host
        key = str(key or "unknown")
        now = time.monotonic()
        cutoff = now - self.policy.rate_limit_window_seconds
        with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.policy.rate_limit_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="请求过于频繁，请稍后再试。",
                )
            bucket.append(now)

        if not self._semaphore.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="服务当前请求较多，请稍后再试。",
            )
        try:
            yield
        finally:
            self._semaphore.release()
