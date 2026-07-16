"""在标准问答链上分派受支持的研究路线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.core.models.query import QueryPlan
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import ResearchRoute, RouteResult


@dataclass
class ResearchRouteService:
    """复用标准复杂问答服务，并提供显式研究路线入口。"""

    standard_service: Any
    graph_route: GraphRoute

    def process(
        self,
        plan: QueryPlan,
        *,
        route: ResearchRoute | str = ResearchRoute.STANDARD,
        trace: dict[str, Any] | None = None,
    ) -> list[Any]:
        active_route = route if isinstance(route, ResearchRoute) else ResearchRoute(route)
        if active_route is ResearchRoute.STANDARD:
            local_trace: dict[str, Any] = {}
            documents = self.standard_service.process(plan, trace=local_trace)
            result = RouteResult(
                documents=documents,
                trace={"route": active_route.value, "base": local_trace},
            )
        elif active_route is ResearchRoute.GRAPH:
            result = self.graph_route.process(lambda: self._run_standard(plan))
        else:
            raise ValueError(f"Unsupported research route: {active_route.value}")
        if trace is not None:
            trace["research_route"] = result.trace
        return result.documents

    def _run_standard(self, plan: QueryPlan) -> RouteResult:
        local_trace: dict[str, Any] = {}
        documents = self.standard_service.process(plan, trace=local_trace)
        return RouteResult(documents=documents, trace=local_trace)
