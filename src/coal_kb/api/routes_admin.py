"""兼容旧管理路由导入路径。"""

from coal_kb.interfaces.api.routes_admin import (
    DocumentInfo,
    IngestResult,
    KBStats,
    build_admin_router,
)

__all__ = ["DocumentInfo", "IngestResult", "KBStats", "build_admin_router"]
