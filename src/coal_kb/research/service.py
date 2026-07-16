"""在标准问答链上分派 Milestone D 研究路线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from coal_kb.core.models.query import QueryPlan
from coal_kb.research.agent import ControlledAgentRoute
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import ResearchRoute, RouteResult
from coal_kb.research.multimodal import MultimodalRoute


@dataclass
class ResearchRouteService:
    """复用标准复杂问答服务并提供统一研究路线入口。"""

    standard_service: Any
    graph_route: GraphRoute
    multimodal_route: MultimodalRoute
    agent_route: ControlledAgentRoute

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
