"""导出摄入注册表协议与 SQLite 实现。"""

from .base import Registry
from .sqlite import RegistrySQLite

__all__ = ["Registry", "RegistrySQLite"]
