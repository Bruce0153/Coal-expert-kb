"""验证受控 Agent 的动作白名单、最大步数和完整 Trace。"""

import pytest
from langchain_core.documents import Document

from coal_kb.research import (
    AgentAction,
    ControlledAgentRoute,
    GraphRoute,
    MultimodalRoute,
    RouteResult,
)


def _base() -> RouteResult:
    return RouteResult(
        documents=[
            Document(
                page_content="steam gasification mechanism",
                metadata={"chunk_id": "c1", "source_file": "a.pdf", "parent_id": "p1"},
            ),
            Document(
                page_content="Figure 2. steam gasification pathway",
                metadata={"chunk_id": "c2", "source_file": "a.pdf", "parent_id": "p1"},
            ),
        ],
        trace={"base": True},
    )


def test_controlled_agent_runs_only_bounded_whitelist_actions() -> None:
    result = ControlledAgentRoute(max_steps=3).process(
        "比较图中的蒸汽气化机制和路径",
        base_route=_base,
        graph_route=GraphRoute(seed_count=1),
        multimodal_route=MultimodalRoute(),
    )

    agent = result.trace["agent"]
    assert [step["action"] for step in agent["steps"]] == [
        "retrieve",
        "graph",
        "multimodal",
    ]
    assert len(agent["steps"]) == 3
    assert agent["allowed_actions"] == ["retrieve", "graph", "multimodal"]
    assert all(step["status"] == "completed" for step in agent["steps"])


def test_controlled_agent_stops_at_max_steps() -> None:
    result = ControlledAgentRoute(max_steps=2).process(
        "比较图中的反应机制",
        base_route=_base,
        graph_route=GraphRoute(),
        multimodal_route=MultimodalRoute(),
    )
    assert len(result.trace["agent"]["steps"]) == 2
    assert result.trace["agent"]["stop_reason"] == "max_steps_reached"


def test_controlled_agent_rejects_disallowed_planned_action() -> None:
    route = ControlledAgentRoute(
        allowed_actions=(AgentAction.RETRIEVE,),
    )
    with pytest.raises(ValueError, match="not allowed"):
        route.process(
            "为什么会产生这种反应机制",
            base_route=_base,
            graph_route=GraphRoute(),
            multimodal_route=MultimodalRoute(),
        )
