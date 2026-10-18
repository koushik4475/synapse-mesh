"""Orchestrator Agent - Coordinates Multi-Agent Mesh Execution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.worker import BaseWorkerAgent, CodeWorker, ResearchWorker, ValidatorWorker
from src.dag.executor import ParallelDispatchEngine, StepExecutor
from src.dag.planner import DAGPlanner, TaskNode
from src.memory.hierarchical import HierarchicalMemoryManager
from src.models import AgentRole, TaskResult, TaskStatus

logger = logging.getLogger("synapse_mesh.orchestrator")


class OrchestratorAgent:
    """Master orchestrator that plans DAG workflows, assigns specialized workers, and controls mesh execution."""

    def __init__(
        self,
        mesh_id: str = "mesh-primary",
        memory: Optional[HierarchicalMemoryManager] = None,
        max_concurrency: int = 8
    ):
        self.mesh_id = mesh_id
        self.memory = memory or HierarchicalMemoryManager()
        self.max_concurrency = max_concurrency
        self.workers: Dict[str, BaseWorkerAgent] = {}

        # Register default worker fleet
        self.register_worker(ResearchWorker(memory=self.memory))
        self.register_worker(CodeWorker(memory=self.memory))
        self.register_worker(ValidatorWorker(memory=self.memory))

    def register_worker(self, worker: BaseWorkerAgent) -> None:
        """Register worker in the agent mesh."""
        self.workers[worker.agent_id] = worker
        # Also index by role name for dynamic routing
        self.workers[worker.role.value] = worker

    def create_step_executor(self) -> StepExecutor:
        """Create step executor bound to registered workers and mesh actions."""
        async def agent_handler(node: TaskNode, context: Dict[str, Any]) -> Any:
            assigned = node.assigned_agent or node.payload.get("role", "CUSTOM")
            worker = self.workers.get(assigned)
            if not worker:
                raise ValueError(f"Worker '{assigned}' not found in mesh registry.")

            return await worker.process_task(node.payload, context)

        async def tool_handler(node: TaskNode, context: Dict[str, Any]) -> Any:
            tool_name = node.payload.get("tool_name")
            args = node.payload.get("arguments", {})
            worker = self.workers.get(AgentRole.RESEARCHER.value)
            if not worker:
                raise ValueError("No worker available to execute tool.")
            return await worker.invoke_tool(tool_name, **args)

        handler_registry = {
            "agent": agent_handler,
            "tool": tool_handler
        }
        return StepExecutor(handler_registry)

    async def execute_plan(
        self,
        planner: DAGPlanner,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, TaskResult]:
        """Dispatches DAG plan through parallel execution mesh."""
        executor = self.create_step_executor()
        engine = ParallelDispatchEngine(
            planner=planner,
            executor=executor,
            max_concurrency=self.max_concurrency,
            shared_context=initial_context or {}
        )

        results = await engine.run()

        # Consolidate results into orchestrator memory fabric
        for task_id, res in results.items():
            self.memory.record_episodic_event(
                content=f"Plan {planner.plan_id} completed task {task_id} with status {res.status.value}",
                metadata={"task_id": task_id, "status": res.status.value, "error": res.error}
            )

        return results
