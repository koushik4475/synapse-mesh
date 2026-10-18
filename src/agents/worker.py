"""Worker Agent Protocol and Specialized Worker Agents."""

from __future__ import annotations

import abc
import asyncio
from typing import Any, Dict, List, Optional

from src.memory.hierarchical import HierarchicalMemoryManager
from src.models import AgentRole, ToolCall
from src.tools.registry import ToolRegistry, default_tool_registry


class BaseWorkerAgent(abc.ABC):
    """Abstract base class for all Synapse Mesh autonomous worker agents."""

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        memory: Optional[HierarchicalMemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.memory = memory or HierarchicalMemoryManager()
        self.tool_registry = tool_registry or default_tool_registry

    @abc.abstractmethod
    async def process_task(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Process incoming task and return computed outcome."""
        pass

    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool via the tool registry."""
        call = ToolCall(tool_name=tool_name, arguments=kwargs)
        resp = await self.tool_registry.invoke(call)
        if not resp.success:
            raise RuntimeError(f"Tool {tool_name} failed: {resp.error}")
        return resp.result


class ResearchWorker(BaseWorkerAgent):
    """Worker specialized in knowledge retrieval, search, and fact extraction."""

    def __init__(
        self,
        agent_id: str = "researcher-01",
        memory: Optional[HierarchicalMemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(agent_id, AgentRole.RESEARCHER, memory, tool_registry)

    async def process_task(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = task_payload.get("query", "")
        # Query external tool or search
        search_res = await self.invoke_tool("web_search", query=query)
        # Check prior memory recall
        prior = self.memory.recall(query, top_k=2)
        recalled_notes = [p[0].content for p in prior]

        summary = f"Research findings for '{query}': {search_res['results'][0]['snippet']}"
        # Record into episodic memory
        self.memory.record_episodic_event(summary, metadata={"query": query})

        return {
            "query": query,
            "summary": summary,
            "recalled_context": recalled_notes,
            "status": "researched"
        }


class CodeWorker(BaseWorkerAgent):
    """Worker specialized in logic generation, algorithmic compute, and code assembly."""

    def __init__(
        self,
        agent_id: str = "coder-01",
        memory: Optional[HierarchicalMemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(agent_id, AgentRole.CODER, memory, tool_registry)

    async def process_task(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        expression = task_payload.get("math_expression")
        calc_result = None
        if expression:
            calc_result = await self.invoke_tool("calculator", expression=expression)

        code_snippet = task_payload.get("code_instruction", "def solve(): return 42")
        self.memory.record_semantic_knowledge(
            f"Code snippet for {expression or code_snippet}: {calc_result}",
            metadata={"type": "code_execution"}
        )

        return {
            "computation": calc_result,
            "generated_code": code_snippet,
            "status": "coded"
        }


class ValidatorWorker(BaseWorkerAgent):
    """Worker specialized in verification, assertions, and invariant checking."""

    def __init__(
        self,
        agent_id: str = "validator-01",
        memory: Optional[HierarchicalMemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(agent_id, AgentRole.VALIDATOR, memory, tool_registry)

    async def process_task(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        upstream = context.get("upstream_results", {})
        assertions = task_payload.get("assertions", [])
        passed = True
        failed_reasons = []

        for assertion in assertions:
            field = assertion.get("field")
            expected_type = assertion.get("expected_type")
            # Verify upstream output has field
            for node_id, out in upstream.items():
                if isinstance(out, dict) and field in out:
                    val = out[field]
                    if expected_type == "float" and not isinstance(val, (int, float)):
                        passed = False
                        failed_reasons.append(f"{field} is not float")
                    elif expected_type == "str" and not isinstance(val, str):
                        passed = False
                        failed_reasons.append(f"{field} is not str")

        return {
            "verified": passed,
            "failed_reasons": failed_reasons,
            "verified_upstream_count": len(upstream)
        }
