"""验证受控 Agent Planner、工具注册表和预算执行器。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from coal_kb.research.agent_models import AgentAction, AgentExecutionBudget, AgentPlanStep
from coal_kb.research.agent_planner import ControlledAgentPlanner
from coal_kb.research.agent_tools import (
    AgentToolContext,
    BudgetedToolExecutor,
    build_controlled_tool_registry,
)
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import RouteResult
from coal_kb.research.multimodal import MultimodalRoute


def _base() -> RouteResult:
    return RouteResult(
        documents=[
            Document(
                page_content="Figure 1 gasification mechanism",
                metadata={"chunk_id": "c1"},
            )
        ],
        trace={},
    )


def _context() -> AgentToolContext:
    return AgentToolContext(
        query="比较图中的气化机制",
        base_route=_base,
        graph_route=GraphRoute(),
        multimodal_route=MultimodalRoute(),
    )


def test_planner_emits_versioned_bounded_plan() -> None:
    planner = ControlledAgentPlanner()
    plan = planner.process(
        "比较图中的气化机制",
        retrieved=_base(),
        multimodal_route=MultimodalRoute(),
        max_steps=2,
    )

    assert [step.action for step in plan.steps] == [AgentAction.RETRIEVE, AgentAction.GRAPH]
    assert plan.truncated is True
    assert plan.to_dict()["version"] == "controlled-agent-plan.v1"


def test_registry_rejects_arbitrary_tools_and_inputs() -> None:
    registry = build_controlled_tool_registry()

    with pytest.raises(ValueError, match="not registered"):
        registry.resolve("shell")
    with pytest.raises(ValueError, match="unsupported inputs"):
        registry.resolve(AgentAction.GRAPH).validate_inputs({"command": "rm -rf /"})


def test_executor_enforces_call_budget() -> None:
    executor = BudgetedToolExecutor(
        registry=build_controlled_tool_registry(),
        budget=AgentExecutionBudget(max_calls=1, max_duration_ms=5000),
    )
    context = _context()
    executor.execute(
        AgentPlanStep(1, AgentAction.RETRIEVE, "required_initial_retrieval"),
        context,
    )

    with pytest.raises(RuntimeError, match="call budget"):
        executor.execute(AgentPlanStep(2, AgentAction.GRAPH, "relationship"), context)


def test_executor_records_failed_tool_calls() -> None:
    executor = BudgetedToolExecutor(
        registry=build_controlled_tool_registry(),
        budget=AgentExecutionBudget(max_calls=2, max_duration_ms=5000),
    )
    context = _context()

    with pytest.raises(ValueError, match="requires retrieved evidence"):
        executor.execute(AgentPlanStep(1, AgentAction.GRAPH, "invalid_order"), context)

    assert executor.records[0].status == "failed"
    assert executor.records[0].error_type == "ValueError"
