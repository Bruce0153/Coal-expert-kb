"""在标准问答链上分派 Milestone D 研究路线。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.core.models.query import QueryPlan
from coal_kb.research.agent import ControlledAgentRoute
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import ResearchRoute, RouteResult
from coal_kb.research.multimodal import MultimodalRoute
from coal_kb.research.visual_assets import VisualAssetIndex


def _configured_multimodal_route() -> MultimodalRoute:
    """仅在显式配置索引路径时加载资产检索。"""

    raw_path = os.getenv("COAL_KB_VISUAL_INDEX_PATH", "").strip()
    if not raw_path:
        return MultimodalRoute()
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configured visual asset index does not exist: {path}")
    top_k = int(os.getenv("COAL_KB_VISUAL_TOP_K", "5"))
    visual_weight = float(os.getenv("COAL_KB_VISUAL_WEIGHT", "1.0"))
    if top_k < 1:
        raise ValueError("COAL_KB_VISUAL_TOP_K must be positive")
    if visual_weight < 0:
        raise ValueError("COAL_KB_VISUAL_WEIGHT cannot be negative")
    return MultimodalRoute(
        visual_index=VisualAssetIndex.load(path),
        visual_top_k=top_k,
        visual_weight=visual_weight,
    )


@dataclass
class ResearchRouteService:
    """复用标准复杂问答服务并提供统一研究路线入口。"""

    standard_service: Any
    graph_route: GraphRoute = field(default_factory=GraphRoute)
    multimodal_route: MultimodalRoute = field(default_factory=_configured_multimodal_route)
    agent_route: ControlledAgentRoute = field(default_factory=ControlledAgentRoute)

    def process(
        self,
        plan: QueryPlan,
        *,
        route: ResearchRoute | str = ResearchRoute.STANDARD,
        trace: dict[str, Any] | None = None,
    ) -> list[Document]:
        active_route = route if isinstance(route, ResearchRoute) else ResearchRoute(route)
        if active_route is ResearchRoute.STANDARD:
            base = self._run_standard(plan)
            result = RouteResult(
                documents=base.documents,
                trace={"route": active_route.value, "base": base.trace},
            )
        elif active_route is ResearchRoute.GRAPH:
            result = self.graph_route.process(lambda: self._run_standard(plan))
        elif active_route is ResearchRoute.MULTIMODAL:
            result = self.multimodal_route.process(
                plan.query.normalized,
                lambda: self._run_standard(plan),
            )
        elif active_route is ResearchRoute.AGENT:
            result = self.agent_route.process(
                plan.query.normalized,
                base_route=lambda: self._run_standard(plan),
                graph_route=self.graph_route,
                multimodal_route=self.multimodal_route,
            )
        else:
            raise ValueError(f"Unsupported research route: {active_route.value}")
        if trace is not None:
            trace["research_route"] = result.trace
        return result.documents

    def _run_standard(self, plan: QueryPlan) -> RouteResult:
        local_trace: dict[str, Any] = {}
        documents = self.standard_service.process(plan, trace=local_trace)
        return RouteResult(documents=list(documents), trace=local_trace)
