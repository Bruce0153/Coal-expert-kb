"""保存进程内运行配置，并为每次请求提供隔离副本。"""

from __future__ import annotations

from threading import RLock

from coal_kb.infra.config import AppConfig


class RuntimeConfigStore:
    """线程安全地保存当前运行配置，不持久化 API Key 到磁盘。"""

    def __init__(self, base_config: AppConfig) -> None:
        self._base = base_config.model_copy(deep=True)
        self._active = base_config.model_copy(deep=True)
        self._lock = RLock()

    def snapshot(self) -> AppConfig:
        """返回当前配置的深拷贝，避免请求之间互相修改。"""
        with self._lock:
            return self._active.model_copy(deep=True)

    def replace(self, config: AppConfig) -> AppConfig:
        """替换当前运行配置并返回新的隔离副本。"""
        with self._lock:
            self._active = config.model_copy(deep=True)
            return self._active.model_copy(deep=True)

    def reset(self) -> AppConfig:
        """恢复服务启动时加载的配置。"""
        with self._lock:
            self._active = self._base.model_copy(deep=True)
            return self._active.model_copy(deep=True)
