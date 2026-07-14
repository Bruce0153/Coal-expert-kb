"""提供保持现有协议的健康状态。"""

from coal_kb.operations import config


def health_status() -> dict[str, str]:
    """返回既有 `/health` 响应。"""
    return {"status": config.HEALTH_STATUS_OK}
