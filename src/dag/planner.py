"""Topological DAG Planner, Task Nodes, and Dependency Graph."""

from __future__ import annotations

import collections
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from src.models import TaskResult, TaskStatus


class TaskNode(BaseModel):
    """Represents an executable unit in the DAG workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    name: str
    action_type: str = "agent"  # agent, tool, function
    assigned_agent: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    dependencies: Set[str] = Field(default_factory=set)
    timeout_seconds: float = 60.0
    retry_count: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None


class DAGValidationError(Exception):
    """Raised when DAG graph contains cycles or invalid nodes."""
    pass


class DAGPlanner:
    """Constructs, validates, and manages execution plans for TaskNodes."""

    def __init__(self, plan_id: str = "default_plan"):
        self.plan_id = plan_id
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[str, Set[str]] = collections.defaultdict(set)  # parent -> children
        self.reverse_edges: Dict[str, Set[str]] = collections.defaultdict(set)  # child -> parents

    def add_node(self, node: TaskNode) -> None:
        """Register a node in the DAG plan."""
        if node.task_id in self.nodes:
            raise DAGValidationError(f"Task node '{node.task_id}' already exists in plan.")
        self.nodes[node.task_id] = node
        # Add pre-configured dependencies if any
        for dep in node.dependencies:
            self.add_edge(dep, node.task_id)

    def add_edge(self, from_task_id: str, to_task_id: str) -> None:
        """Add directed dependency: `from_task_id` must finish before `to_task_id`."""
        if from_task_id not in self.nodes:
            raise DAGValidationError(f"Source task '{from_task_id}' not found.")
        if to_task_id not in self.nodes:
            raise DAGValidationError(f"Destination task '{to_task_id}' not found.")

        self.edges[from_task_id].add(to_task_id)
        self.reverse_edges[to_task_id].add(from_task_id)
        self.nodes[to_task_id].dependencies.add(from_task_id)

    def validate(self) -> None:
        """Validate DAG acyclicity using Kahn's algorithm."""
        in_degree: Dict[str, int] = {k: len(self.reverse_edges[k]) for k in self.nodes}
        queue = collections.deque([k for k, d in in_degree.items() if d == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for nxt in self.edges[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited_count != len(self.nodes):
            raise DAGValidationError("Cycle detected in DAG execution graph.")

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[TaskNode]:
        """Return tasks whose dependencies are all satisfied and status is PENDING."""
        ready: List[TaskNode] = []
        for task_id, node in self.nodes.items():
            if node.status == TaskStatus.PENDING:
                # Check if all dependencies are in completed_tasks
                if node.dependencies.issubset(completed_tasks):
                    ready.append(node)
        return ready

    def get_topological_order(self) -> List[str]:
        """Compute strict topological order of node IDs."""
        self.validate()
        in_degree: Dict[str, int] = {k: len(self.reverse_edges[k]) for k in self.nodes}
        queue = collections.deque(sorted([k for k, d in in_degree.items() if d == 0]))
        order: List[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for nxt in sorted(self.edges[curr]):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return order
