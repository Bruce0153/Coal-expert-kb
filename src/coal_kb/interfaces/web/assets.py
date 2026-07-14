"""解析当前兼容网页静态资源目录。"""

from pathlib import Path

from coal_kb.interfaces.web import config


def web_static_dir() -> Path:
    """返回网页静态资源目录，保持现有打包路径不变。"""
    package_root = Path(__file__).resolve().parents[2]
    return package_root.joinpath(*config.STATIC_RELATIVE_PATH)
