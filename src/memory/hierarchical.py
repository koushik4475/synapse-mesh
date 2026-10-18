"""Hierarchical Memory Fabric with Working, Episodic, and Semantic tiers."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.memory.vector_store import VectorStore
from src.models import MemoryEntry


def deterministic_embedding(text: str, dimension: int = 128) -> List[float]:
    """Generate high-entropy deterministic pseudo-embedding from text for testing & offline usage."""
    words = text.lower().strip().split()
    vec = np.zeros(dimension, dtype=np.float32)

    if not words:
        vec[0] = 1.0
        return vec.tolist()

    for idx, word in enumerate(words):
        h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
        # Seed indices and values deterministically
        bucket = h % dimension
        weight = 1.0 / (1.0 + 0.1 * idx)
        vec[bucket] += float(weight * ((h % 100) / 100.0 + 0.1))

    # Normalize vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class HierarchicalMemoryManager:
    """Manages multi-tiered agent memory: Working Memory, Episodic Memory, Semantic Long-Term."""

    def __init__(self, dimension: int = 128, decay_half_life_seconds: float = 3600.0):
        self.dimension = dimension
        self.decay_half_life = decay_half_life_seconds

        # Memory tiers
        self.working_memory: Dict[str, Any] = {}
        self.episodic_store = VectorStore(dimension=dimension)
        self.semantic_store = VectorStore(dimension=dimension)

    def write_working_memory(self, key: str, value: Any) -> None:
        """Store transient scratchpad state for active execution."""
        self.working_memory[key] = value

    def read_working_memory(self, key: str, default: Any = None) -> Any:
        """Read transient state."""
        return self.working_memory.get(key, default)

    def clear_working_memory(self) -> None:
        """Reset active scratchpad."""
        self.working_memory.clear()

    def record_episodic_event(
        self,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Record task event or agent interaction into episodic memory."""
        vec = vector or deterministic_embedding(content, self.dimension)
        meta = metadata or {}
        entry = MemoryEntry(
            content=content,
            vector=vec,
            metadata=meta,
            tier="episodic",
            timestamp=time.time()
        )
        self.episodic_store.add_entry(entry)
        return entry

    def record_semantic_knowledge(
        self,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Record consolidated durable knowledge into semantic long-term memory."""
        vec = vector or deterministic_embedding(content, self.dimension)
        meta = metadata or {}
        entry = MemoryEntry(
            content=content,
            vector=vec,
            metadata=meta,
            tier="semantic",
            timestamp=time.time()
        )
        self.semantic_store.add_entry(entry)
        return entry

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.2,
        query_vector: Optional[List[float]] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """Unified recall across episodic and semantic memory fabric."""
        vec = query_vector or deterministic_embedding(query, self.dimension)

        episodic_hits = self.episodic_store.search(vec, top_k=top_k, min_similarity=min_similarity)
        semantic_hits = self.semantic_store.search(vec, top_k=top_k, min_similarity=min_similarity)

        # Merge and score with recency decay weighting
        now = time.time()
        combined: List[Tuple[MemoryEntry, float]] = []

        for entry, sim in episodic_hits + semantic_hits:
            age = max(0.0, now - entry.timestamp)
            # Exponential decay: e^(-0.693 * age / half_life)
            recency_factor = np.exp(-0.693 * (age / self.decay_half_life)) if entry.tier == "episodic" else 1.0
            adjusted_score = float(sim * (0.8 + 0.2 * recency_factor))
            combined.append((entry, adjusted_score))

        # Sort combined
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
