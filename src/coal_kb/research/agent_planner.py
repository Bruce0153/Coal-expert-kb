"""生成并验证受控 Agent 的结构化计划。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from coal_kb.research.agent_models import AgentAction, AgentPlan, AgentPlanStep
from coal_kb.research.models import RouteResult
from coal_kb.research.multimodal import MultimodalRoute

_RELATION_QUERY = re.compile(
    r"比较|关系|机制|为什么|导致|路径|compare|relationship|mechanism|why|cause|pathway",
    re.IGNORECASE,
)


@dataclass
class ControlledAgentPlanner:
    """仅从固定动作集合生成确定性计划。"""

    def process(
        self,
        query: str,
        *,
        retrieved: RouteResult,
        multimodal_route: MultimodalRoute,
        max_steps: int,
    ) -> AgentPlan:
        if max_steps < 1:
            raise ValueError("Controlled agent max_steps must be positive")
        candidates: list[tuple[AgentAction, str]] = [
            (AgentAction.RETRIEVE, "required_initial_retrieval")
        ]
        if _RELATION_QUERY.search(query):
            candidates.append((AgentAction.GRAPH, "relationship_or_mechanism_query"))
        requested = multimodal_route.requested_modalities(query)
        available = {
            multimodal_route.infer_modality(document)
            for document in retrieved.documents
        }
        if requested != {"text"} or available - {"text"}:
            candidates.append((AgentAction.MULTIMODAL, "multimodal_evidence_available_or_requested"))
        truncated = len(candidates) > max_steps
        selected = candidates[:max_steps]
        plan = AgentPlan(
            query=query,
            steps=tuple(
                AgentPlanStep(index=index, action=action, reason=reason)
                for index, (action, reason) in enumerate(selected, start=1)
            ),
            truncated=truncated,
        )
        plan.validate()
        return plan

    @staticmethod
    def validate_allowed(plan: AgentPlan, allowed_actions: tuple[AgentAction, ...]) -> None:
        allowed = set(allowed_actions)
        for step in plan.steps:
            if step.action not in allowed:
                raise ValueError(f"Agent action is not allowed: {step.action.value}")
