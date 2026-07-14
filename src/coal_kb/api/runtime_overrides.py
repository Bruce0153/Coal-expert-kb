"""兼容旧请求配置覆盖导入路径。"""

from coal_kb.interfaces.api.runtime_overrides import (
    apply_runtime_overrides,
    build_settings_defaults,
)

__all__ = ["apply_runtime_overrides", "build_settings_defaults"]
