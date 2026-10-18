"""Tests for Vector Store and Hierarchical Memory Fabric."""

import pytest
import numpy as np
from src.memory.vector_store import VectorStore
from src.memory.hierarchical import HierarchicalMemoryManager, deterministic_embedding
from src.models import MemoryEntry


def test_vector_store_add_and_search():
    store = VectorStore(dimension=4)
    e1 = MemoryEntry(content="doc1", vector=[1.0, 0.0, 0.0, 0.0])
    e2 = MemoryEntry(content="doc2", vector=[0.0, 1.0, 0.0, 0.0])
    e3 = MemoryEntry(content="doc3", vector=[0.707, 0.707, 0.0, 0.0])

    store.add_entry(e1)
    store.add_entry(e2)
    store.add_entry(e3)

    # Search with vector close to e1
    results = store.search([1.0, 0.1, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0].content == "doc1"
    assert results[0][1] > 0.9
    assert results[0][0].access_count == 1


def test_vector_store_dimension_mismatch():
    store = VectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add_entry(MemoryEntry(content="fail", vector=[1.0, 0.0]))

    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0])


def test_hierarchical_memory_tiers():
    mem = HierarchicalMemoryManager(dimension=16)

    # Working memory test
    mem.write_working_memory("active_task", "analysis_001")
    assert mem.read_working_memory("active_task") == "analysis_001"
    mem.clear_working_memory()
    assert mem.read_working_memory("active_task") is None

    # Episodic and Semantic recall
    entry_ep = mem.record_episodic_event("Executed database migration for shard 4")
    entry_sem = mem.record_semantic_knowledge("Production database utilizes 8 shards with raft consensus")

    assert entry_ep.tier == "episodic"
    assert entry_sem.tier == "semantic"

    # Recall query
    recalled = mem.recall("database migration shards", top_k=2)
    assert len(recalled) > 0
    contents = [r[0].content for r in recalled]
    assert any("database migration" in c for c in contents)
