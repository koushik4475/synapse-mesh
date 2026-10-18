"""Core types and data models for Synapse Mesh."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Execution status of a task node in DAG."""
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class AgentRole(str, Enum):
    """Specialized worker/orchestrator agent roles."""
    ORCHESTRATOR = "ORCHESTRATOR"
    RESEARCHER = "RESEARCHER"
    CODER = "CODER"
    VALIDATOR = "VALIDATOR"
    SYNTHESIZER = "SYNTHESIZER"
    CUSTOM = "CUSTOM"


class TaskResult(BaseModel):
    """Output payload from an executed task."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class MemoryEntry(BaseModel):
    """Vector fabric memory entry with hierarchical scope."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tier: str = "working"  # "working", "short_term", "long_term"
    timestamp: float = Field(default_factory=time.time)
    access_count: int = 0
    decay_weight: float = 1.0


class ToolDefinition(BaseModel):
    """Specification of an invocable agent tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Request by an agent to execute a tool."""
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    """Result of a tool execution."""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
