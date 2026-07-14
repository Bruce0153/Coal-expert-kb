"""规范化上传文件名并保持现有重复文件重命名行为。"""

from __future__ import annotations

import uuid
from pathlib import Path

from coal_kb.infra.security import config


def safe_upload_name(filename: str) -> str:
    """移除客户端路径部分，只保留文件名。"""
    return Path(filename).name


def build_upload_path(directory: Path, filename: str) -> Path:
    """生成目标路径；重名时追加六位 UUID 后缀。"""
    safe_name = safe_upload_name(filename)
    destination = directory / safe_name
    if destination.exists():
        destination = destination.with_name(
            f"{destination.stem}_{uuid.uuid4().hex[: config.UPLOAD_SUFFIX_LENGTH]}{destination.suffix.lower()}"
        )
    return destination
