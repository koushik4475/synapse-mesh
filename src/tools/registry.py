"""Extensible Tool Execution Registry."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Coroutine, Dict, Optional

from src.models import ToolCall, ToolDefinition, ToolResponse


class ToolRegistry:
    """Registry for registering and invoking tools safely."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Decorator to register a tool implementation."""
        def decorator(fn: Callable) -> Callable:
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters_schema=parameters_schema or {}
            )
            self._tools[name] = tool_def
            self._handlers[name] = fn
            return fn
        return decorator

    def register_function(
        self,
        name: str,
        description: str,
        fn: Callable,
        parameters_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Direct registration of a function as a tool."""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema or {}
        )
        self._tools[name] = tool_def
        self._handlers[name] = fn

    def list_tools(self) -> Dict[str, ToolDefinition]:
        """Returns all registered tool specifications."""
        return dict(self._tools)

    async def invoke(self, call: ToolCall) -> ToolResponse:
        """Invoke registered tool with supplied arguments."""
        if call.tool_name not in self._handlers:
            return ToolResponse(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=f"Tool '{call.tool_name}' is not registered in ToolRegistry."
            )

        handler = self._handlers[call.tool_name]
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(**call.arguments)
            else:
                result = handler(**call.arguments)

            return ToolResponse(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                result=result
            )
        except Exception as exc:
            return ToolResponse(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error=str(exc)
            )


# Default shared registry
default_tool_registry = ToolRegistry()


@default_tool_registry.register(
    name="calculator",
    description="Safely computes basic mathematical expressions.",
    parameters_schema={"expression": "string"}
)
def compute_math(expression: str) -> float:
    """Safe evaluation of arithmetic expressions."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        raise ValueError("Invalid characters in expression.")
    return float(eval(expression, {"__builtins__": {}}, {}))


@default_tool_registry.register(
    name="web_search",
    description="Simulates web search query lookup for agent research.",
    parameters_schema={"query": "string"}
)
def search_sim(query: str) -> Dict[str, Any]:
    """Search simulation returning relevant contextual snippets."""
    return {
        "query": query,
        "results": [
            {"title": f"Result for {query}", "snippet": f"Synthetic evidence for {query}."}
        ]
    }
