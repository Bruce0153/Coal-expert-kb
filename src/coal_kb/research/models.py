"""定义研究路线、执行结果和实验配置协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class ResearchRoute(str, Enum):
    """Milestone D 当前可执行的研究路线。"""

    STANDARD = "standard"
    GRAPH = "graph"


@dataclass(frozen=True)
class RouteResult:
    """研究路线统一返回标准文档和可审计 Trace。"""

    documents: list[Document]
    trace: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    """定义一次可复现研究实验的固定输入。"""

    name: str
    route: ResearchRoute
    dataset_path: Path
    output_dir: Path
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    metadata: dict[str, Any] = field(default_factory=dict)
