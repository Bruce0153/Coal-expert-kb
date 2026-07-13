"""兼容旧导入路径，实际实现位于 coal_kb.infra.persistence.search.elasticsearch。"""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module("coal_kb.infra.persistence.search.elasticsearch")
sys.modules[__name__] = _implementation
