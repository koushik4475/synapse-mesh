"""Step Executor and Parallel Dispatch Engine for Synapse Mesh."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Set

from src.dag.planner import DAGPlanner, TaskNode
from src.models import TaskResult, TaskStatus

logger = logging.getLogger("synapse_mesh.executor")


class StepExecutor:
    """Executes single task nodes with retry policy and telemetry tracking."""

    def __init__(self, handler_registry: Dict[str, Callable[[TaskNode, Dict[str, Any]], Coroutine[Any, Any, Any]]]):
        self.handler_registry = handler_registry

    async def execute_step(self, node: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """Executes a single node, invoking associated agent or function handler."""
        node.status = TaskStatus.RUNNING
        start_time = time.perf_counter()
        retries_left = node.retry_count
        last_error: Optional[str] = None

        while retries_left >= 0:
            try:
                # Look up handler
                handler = self.handler_registry.get(node.action_type)
                if not handler:
                    raise ValueError(f"No handler registered for action type: '{node.action_type}'")

                # Run with timeout
                result_output = await asyncio.wait_for(
                    handler(node, context),
                    timeout=node.timeout_seconds
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                result = TaskResult(
                    task_id=node.task_id,
                    status=TaskStatus.COMPLETED,
                    output=result_output,
                    execution_time_ms=elapsed_ms,
                    metadata={"retries_remaining": retries_left}
                )
                node.status = TaskStatus.COMPLETED
                node.result = result
                return result

            except Exception as exc:
                last_error = str(exc)
                retries_left -= 1
                logger.warning(
                    f"Task {node.task_id} failed: {last_error}. Retries left: {max(retries_left, 0)}"
                )
                if retries_left >= 0:
                    await asyncio.sleep(0.05 * (node.retry_count - retries_left + 1))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        result = TaskResult(
            task_id=node.task_id,
            status=TaskStatus.FAILED,
            error=last_error,
            execution_time_ms=elapsed_ms
        )
        node.status = TaskStatus.FAILED
        node.result = result
        return result


class ParallelDispatchEngine:
    """Coordinates parallel topological execution of DAG with concurrency limits."""

    def __init__(
        self,
        planner: DAGPlanner,
        executor: StepExecutor,
        max_concurrency: int = 8,
        shared_context: Optional[Dict[str, Any]] = None
    ):
        self.planner = planner
        self.executor = executor
        self.max_concurrency = max_concurrency
        self.shared_context = shared_context or {}
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self) -> Dict[str, TaskResult]:
        """Executes the full DAG graph asynchronously to completion."""
        self.planner.validate()
        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        results: Dict[str, TaskResult] = {}
        active_futures: Dict[asyncio.Task, str] = {}

        total_nodes = len(self.planner.nodes)

        async def _run_guarded(task_node: TaskNode) -> TaskResult:
            async with self.semaphore:
                # Merge outputs of completed dependency tasks into step context
                step_ctx = dict(self.shared_context)
                step_ctx["upstream_results"] = {
                    dep_id: results[dep_id].output for dep_id in task_node.dependencies if dep_id in results
                }
                return await self.executor.execute_step(task_node, step_ctx)

        while len(completed_tasks) + len(failed_tasks) < total_nodes:
            # 1. Schedule ready tasks
            ready_nodes = self.planner.get_ready_tasks(completed_tasks)
            for node in ready_nodes:
                # Check if any dependencies failed - if so, fail/skip task
                dep_failed = any(dep in failed_tasks for dep in node.dependencies)
                if dep_failed:
                    node.status = TaskStatus.SKIPPED
                    res = TaskResult(
                        task_id=node.task_id,
                        status=TaskStatus.SKIPPED,
                        error="Upstream dependency failed."
                    )
                    node.result = res
                    results[node.task_id] = res
                    failed_tasks.add(node.task_id)
                    continue

                # Schedule execution
                node.status = TaskStatus.RUNNING
                fut = asyncio.create_task(_run_guarded(node))
                active_futures[fut] = node.task_id

            if not active_futures:
                # If no active tasks and uncompleted tasks remain, deadlock or unmet dependencies
                break

            # 2. Wait for at least one active task to complete
            done, _ = await asyncio.wait(
                active_futures.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )

            for fut in done:
                task_id = active_futures.pop(fut)
                res: TaskResult = await fut
                results[task_id] = res
                if res.status == TaskStatus.COMPLETED:
                    completed_tasks.add(task_id)
                else:
                    failed_tasks.add(task_id)

        return results
