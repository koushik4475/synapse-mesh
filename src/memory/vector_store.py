"""High-performance Cosine Similarity Vector Store."""

from __future__ import annotations

import heapq
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.models import MemoryEntry


class VectorStore:
    """In-memory numpy vector store supporting cosine similarity and top-k retrieval."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.entries: Dict[str, MemoryEntry] = {}
        self._matrix: Optional[np.ndarray] = None
        self._id_index: List[str] = []
        self._dirty: bool = False

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add or update a vector memory entry."""
        if len(entry.vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, got {len(entry.vector)}"
            )
        self.entries[entry.entry_id] = entry
        self._dirty = True

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self._dirty = True
            return True
        return False

    def _sync_matrix(self) -> None:
        """Synchronize dense numpy array and normalize vectors for cosine similarity."""
        if not self._dirty and self._matrix is not None:
            return

        if not self.entries:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            self._id_index = []
            self._dirty = False
            return

        self._id_index = list(self.entries.keys())
        raw_vectors = np.array([self.entries[k].vector for k in self._id_index], dtype=np.float32)

        # Normalize rows to unit vectors for fast dot product cosine similarity
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = raw_vectors / norms
        self._dirty = False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_fn: Optional[Any] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """Perform cosine similarity top-k search against query vector."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, got {len(query_vector)}"
            )

        if not self.entries:
            return []

        self._sync_matrix()
        assert self._matrix is not None

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1e-10
        q_unit = q / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self._matrix, q_unit)

        results: List[Tuple[MemoryEntry, float]] = []
        for idx, score in enumerate(scores):
            score_val = float(score)
            if score_val < min_similarity:
                continue

            entry_id = self._id_index[idx]
            entry = self.entries[entry_id]

            if filter_fn and not filter_fn(entry):
                continue

            results.append((entry, score_val))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        # Update access telemetry
        for entry, _ in top_results:
            entry.access_count += 1

        return top_results
