<div align="center">

# ⚡ Synapse Mesh

**Multi-Agent Autonomous LLM Execution Mesh with DAG Scheduling & Memory Vector Fabric**

[![CI](https://github.com/ER-brocode/synapse-mesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ER-brocode/synapse-mesh/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

```
        ┌────────────────────────────────────────────────────────┐
        │             SYNAPSE ORCHESTRATOR MESH                  │
        └─────────────────────────┬──────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │  RESEARCHER  │         │  CODER / AGT │         │  VALIDATOR   │
  │    AGENT     │         │   WORKER     │         │    AGENT     │
  └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │    HIERARCHICAL VECTOR FABRIC │
                  │  Working ⇄ Episodic ⇄ Semantic│
                  └───────────────────────────────┘
```

</div>

---

## 🌟 Overview

**Synapse Mesh** is a distributed multi-agent execution framework engineered for deterministic, high-throughput LLM workflow orchestration. By coupling **Directed Acyclic Graph (DAG) topological scheduling** with an embedded **Hierarchical Memory Vector Fabric**, Synapse Mesh eliminates hallucination cascades, guarantees strict dependency execution order, and empowers autonomous worker agents with persistent semantic recall.

---

## ✨ Key Capabilities

- 🕸️ **Topological DAG Planner**: Deterministic cycle detection using Kahn's algorithm, automated dependency resolution, and dynamic parallel dispatching.
- ⚡ **Parallel Concurrency Engine**: Asynchronous non-blocking execution with configurable semaphore limits, fault isolation, and auto-retry backoff policies.
- 🧠 **Hierarchical Memory Vector Fabric**:
  - **Working Memory**: Instantaneous thread-safe volatile scratchpad for active reasoning.
  - **Episodic Memory**: Event-indexed trajectory logs with exponential temporal recency decay.
  - **Semantic Long-Term Store**: Dense cosine similarity vector store with fast NumPy matrix dot-product retrieval.
- 🤖 **Autonomous Worker Protocols**: Standardized agent abstractions for Researchers, Coders, Validators, and custom extensible domain agents.
- 🛠️ **Pluggable Tool Registry**: Schema-validated tool dispatch supporting arithmetic computation, web query simulators, and asynchronous external APIs.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Orchestration ["Mesh Orchestrator Engine"]
        DAG[Topological DAG Planner] --> Scheduler[Parallel Dispatch Engine]
        Scheduler --> WorkerPool[Worker Agent Fleet]
    end

    subgraph Agents ["Specialized Agents"]
        WorkerPool --> A1[Researcher Agent]
        WorkerPool --> A2[Coder Agent]
        WorkerPool --> A3[Validator Agent]
    end

    subgraph Memory ["Hierarchical Memory Fabric"]
        WM[Working Scratchpad]
        EM[Episodic Event Logs]
        SM[Semantic Vector Store]
        A1 <--> Memory
        A2 <--> Memory
        A3 <--> Memory
    end

    subgraph Tools ["Tool Registry"]
        A1 --> T1[Web Search]
        A2 --> T2[Math Evaluator]
    end
```

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/ER-brocode/synapse-mesh.git
cd synapse-mesh
pip install -e .
```

### Basic Usage

```python
import asyncio
from src.dag.planner import DAGPlanner, TaskNode
from src.agents.orchestrator import OrchestratorAgent
from src.models import AgentRole

async def main():
    # Initialize orchestrator mesh
    orchestrator = OrchestratorAgent(max_concurrency=4)
    planner = DAGPlanner(plan_id="analytics_workflow")

    # Define tasks
    t1 = TaskNode(
        task_id="research_task",
        name="Research topic",
        action_type="agent",
        assigned_agent=AgentRole.RESEARCHER.value,
        payload={"query": "distributed consensus algorithms"}
    )

    t2 = TaskNode(
        task_id="compute_task",
        name="Run algorithm complexity calculations",
        action_type="agent",
        assigned_agent=AgentRole.CODER.value,
        payload={"math_expression": "2 ** 10 + 24"}
    )

    t3 = TaskNode(
        task_id="validator_task",
        name="Validate workflow outputs",
        action_type="agent",
        assigned_agent=AgentRole.VALIDATOR.value,
        payload={"assertions": [{"field": "computation", "expected_type": "float"}]},
        dependencies={"research_task", "compute_task"}
    )

    planner.add_node(t1)
    planner.add_node(t2)
    planner.add_node(t3)

    # Execute mesh DAG
    results = await orchestrator.execute_plan(planner)
    for task_id, res in results.items():
        print(f"[{res.status.value}] {task_id} -> {res.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 Running Tests

The test suite validates DAG topological order, cycle detection, multi-step concurrent execution, and cosine similarity vector recall:

```bash
python -m pytest -v
```

---

## 📂 Project Structure

```
synapse-mesh/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI matrix pipeline
├── src/
│   ├── dag/
│   │   ├── __init__.py
│   │   ├── planner.py          # Topological DAG planner & cycle detector
│   │   └── executor.py         # Step executor & parallel dispatch engine
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # Cosine similarity vector engine
│   │   └── hierarchical.py     # 3-tier memory fabric manager
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── worker.py           # BaseWorkerAgent, Researcher, Coder, Validator
│   │   └── orchestrator.py     # Orchestrator agent mesh coordinator
│   ├── tools/
│   │   ├── __init__.py
│   │   └── registry.py         # Extensible tool execution registry
│   └── models.py               # Pydantic data schemas
├── tests/
│   ├── __init__.py
│   ├── test_dag_execution.py   # Multi-agent DAG execution verification
│   └── test_memory.py          # Vector store & hierarchical memory tests
├── Cargo.toml                  # Rust crate metadata
├── package.json                # NPM manifest
├── pyproject.toml              # Python build configuration
└── README.md                   # System documentation
```

---

## 🛡️ License

Apache License 2.0. Author: **koushik4475** (`koushik4475@gmail.com`).
