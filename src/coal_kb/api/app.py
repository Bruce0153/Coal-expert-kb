"""兼容旧 FastAPI 应用导入路径。"""

from coal_kb.interfaces.api.app import app, create_app

__all__ = ["app", "create_app"]
