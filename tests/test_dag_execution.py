"""Tests for Multi-Step DAG Agent Execution."""

import pytest
import asyncio
from src.dag.planner import DAGPlanner, DAGValidationError, TaskNode
from src.agents.orchestrator import OrchestratorAgent
from src.models import TaskStatus, AgentRole


def test_dag_cycle_detection():
    planner = DAGPlanner()
    t1 = TaskNode(task_id="t1", name="Task 1")
    t2 = TaskNode(task_id="t2", name="Task 2")
    planner.add_node(t1)
    planner.add_node(t2)

    planner.add_edge("t1", "t2")
    planner.add_edge("t2", "t1")

    with pytest.raises(DAGValidationError):
        planner.validate()


def test_multi_step_dag_agent_execution():
    async def _run():
        orchestrator = OrchestratorAgent(max_concurrency=4)
        planner = DAGPlanner(plan_id="data_pipeline_alpha")

        # Step 1: Research query
        research_task = TaskNode(
            task_id="step_1_research",
            name="Research Data Query",
            action_type="agent",
            assigned_agent=AgentRole.RESEARCHER.value,
            payload={"query": "quantum computing algorithms"}
        )

        # Step 2: Math and code computation
        compute_task = TaskNode(
            task_id="step_2_compute",
            name="Compute complexity factor",
            action_type="agent",
            assigned_agent=AgentRole.CODER.value,
            payload={"math_expression": "128 * 4 + 50", "code_instruction": "def q_eval(): return 562"}
        )

        # Step 3: Tool Invocation Task (direct tool execution)
        calc_task = TaskNode(
            task_id="step_3_tool_eval",
            name="Evaluate direct tool",
            action_type="tool",
            payload={"tool_name": "calculator", "arguments": {"expression": "25 * 4"}}
        )

        # Step 4: Validation step depending on compute & tool tasks
        validation_task = TaskNode(
            task_id="step_4_validate",
            name="Validate results",
            action_type="agent",
            assigned_agent=AgentRole.VALIDATOR.value,
            payload={
                "assertions": [
                    {"field": "computation", "expected_type": "float"},
                    {"field": "summary", "expected_type": "str"}
                ]
            },
            dependencies={"step_1_research", "step_2_compute"}
        )

        planner.add_node(research_task)
        planner.add_node(compute_task)
        planner.add_node(calc_task)
        planner.add_node(validation_task)

        results = await orchestrator.execute_plan(planner)

        assert len(results) == 4
        for tid, res in results.items():
            assert res.status == TaskStatus.COMPLETED, f"Task {tid} failed: {res.error}"

        # Verify research step output
        assert results["step_1_research"].output["status"] == "researched"
        assert "summary" in results["step_1_research"].output

        # Verify compute output
        assert results["step_2_compute"].output["computation"] == 562.0

        # Verify tool task output
        assert results["step_3_tool_eval"].output == 100.0

        # Verify validator output
        assert results["step_4_validate"].output["verified"] is True
        assert results["step_4_validate"].output["verified_upstream_count"] == 2

    asyncio.run(_run())
