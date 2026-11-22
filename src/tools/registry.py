"""Extensible Tool Execution Registry."""

from __future__ import annotations

import ast
import inspect
import math
import operator
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


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    """Evaluate a deliberately small arithmetic AST without executing Python code."""
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
    else:
        raise ValueError("Only basic arithmetic expressions are allowed.")

    if isinstance(value, int) and value.bit_length() > 1024:
        raise ValueError("Result is too large.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite.")
    return value


@default_tool_registry.register(
    name="calculator",
    description="Safely computes bounded basic mathematical expressions.",
    parameters_schema={"expression": "string"}
)
def compute_math(expression: str) -> float:
    """Evaluate bounded arithmetic without Python eval or arbitrary code execution."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    if len(expression) > 256:
        raise ValueError("Expression is too long.")

    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid arithmetic expression.") from exc

    if sum(1 for _ in ast.walk(parsed)) > 64:
        raise ValueError("Expression is too complex.")

    return float(_evaluate_arithmetic(parsed))


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
