"""
ChunkStore — stores and queries chunk-level embedding vectors for the
personal knowledge cache.

Two backends selected at init time:
  disk   — when wiki_save_dir is provided (str path)
  memory — when wiki_save_dir is None

Chunk format (one JSON object per line in chunks.jsonl):
{
    "chunk_index": 0,           # int, position within source document
    "text":        "...",       # str, the chunk's text content
    "vector":      [0.02, ...], # list[float], the embedding vector
    "section":     "3.2",       # str, section heading or "" if absent
    "hit_count":   0,           # int, times this chunk matched a query
    "last_hit_at": null         # str ISO timestamp or null
}
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def cosine_similarity_matrix(
    query_vec: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between query_vec and every row of matrix.
    Safe against zero-norm vectors (returns 0.0 for those).
    Returns array of shape (n_chunks,) with values in [-1, 1].
    """
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    q = query_vec / query_norm

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)  # avoid div by zero
    m = matrix / row_norms

    return q @ m.T


class ChunkStore:
    """
    Stores and retrieves chunk-level embedding vectors for cached documents.

    Disk backend: chunks stored as JSONL files under
        <wiki_save_dir>/<user_id>/<doc_id>/chunks.jsonl
    Memory backend: stored in a plain dict (lost on process exit).
    """

    def __init__(self, wiki_save_dir: Optional[str] = None) -> None:
        self._dir  = Path(wiki_save_dir) if wiki_save_dir else None
        self._mem: dict[tuple[str, str], list[dict]] = {}
        self._lock = threading.Lock()

    # ─── Path helper ──────────────────────────────────────────────────────────

    def _chunk_path(self, user_id: str, doc_id: str) -> Path:
        """Returns the path to chunks.jsonl. Only valid when self._dir is set."""
        assert self._dir is not None
        # Sanitise doc_id for use as a directory name
        safe_doc_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
        safe_user_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in user_id)
        return self._dir / safe_user_id / safe_doc_id / "chunks.jsonl"

    # ─── Core I/O ─────────────────────────────────────────────────────────────

    def load_chunks(self, user_id: str, doc_id: str) -> list[dict]:
        """Load all chunks for a user+doc pair. Returns [] if none stored."""
        with self._lock:
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                if not path.exists():
                    return []
                chunks = []
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            chunks.append(json.loads(line))
                return chunks
            else:
                import copy
                return copy.deepcopy(self._mem.get((user_id, doc_id), []))

    def save_chunks(self, user_id: str, doc_id: str, chunks: list[dict]) -> None:
        """Persist chunks. Atomic on disk (write to .tmp then rename)."""
        with self._lock:
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, separators=(",", ":")) + "\n")
                tmp.rename(path)
            else:
                import copy
                self._mem[(user_id, doc_id)] = copy.deepcopy(chunks)

    def add_chunks(self, user_id: str, doc_id: str, new_chunks: list[dict]) -> int:
        """
        Merge new_chunks into existing chunks, deduplicating by chunk_index.
        Returns count of chunks actually added (0 if all duplicates).
        Thread-safe — acquires lock once for the full load+merge+save.
        """
        with self._lock:
            # Load without re-acquiring lock
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                existing: list[dict] = []
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                existing.append(json.loads(line))
            else:
                import copy
                existing = copy.deepcopy(self._mem.get((user_id, doc_id), []))

            existing_indices = {c["chunk_index"] for c in existing}
            to_add = [
                c for c in new_chunks
                if c["chunk_index"] not in existing_indices
            ]

            if not to_add:
                return 0

            # Ensure defaults
            for c in to_add:
                c.setdefault("hit_count", 0)
                c.setdefault("last_hit_at", None)

            merged = existing + to_add

            # Save without re-acquiring lock (already held)
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for chunk in merged:
                        f.write(json.dumps(chunk, separators=(",", ":")) + "\n")
                tmp.rename(path)
            else:
                import copy
                self._mem[(user_id, doc_id)] = copy.deepcopy(merged)

            return len(to_add)

    def record_hits(
        self,
        user_id: str,
        doc_id: str,
        hit_indices: list[int],
        now: datetime,
    ) -> None:
        """Increment hit_count and stamp last_hit_at for matched chunks."""
        with self._lock:
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                if not path.exists():
                    return
                chunks = []
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            chunks.append(json.loads(line))
            else:
                import copy
                chunks = copy.deepcopy(self._mem.get((user_id, doc_id), []))

            hit_set = set(hit_indices)
            for chunk in chunks:
                if chunk["chunk_index"] in hit_set:
                    chunk["hit_count"] = chunk.get("hit_count", 0) + 1
                    chunk["last_hit_at"] = now.isoformat()

            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                tmp = path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, separators=(",", ":")) + "\n")
                tmp.rename(path)
            else:
                import copy
                self._mem[(user_id, doc_id)] = copy.deepcopy(chunks)

    def delete(self, user_id: str, doc_id: str) -> None:
        """Remove all chunks for a user+doc pair."""
        with self._lock:
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                if path.exists():
                    path.unlink()
                # Remove parent dir if empty
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            else:
                self._mem.pop((user_id, doc_id), None)

    def get_hit_rate(self, user_id: str, doc_id: str) -> float:
        """Fraction of stored chunks that have been hit at least once."""
        with self._lock:
            if self._dir is not None:
                path = self._chunk_path(user_id, doc_id)
                if not path.exists():
                    return 0.0
                chunks = []
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            chunks.append(json.loads(line))
            else:
                import copy
                chunks = copy.deepcopy(self._mem.get((user_id, doc_id), []))

        if not chunks:
            return 0.0
        return len([c for c in chunks if c.get("hit_count", 0) > 0]) / len(chunks)

    # ─── Static helpers ───────────────────────────────────────────────────────

    @staticmethod
    def build_matrix(chunks: list[dict]) -> Optional[np.ndarray]:
        """
        Stack chunk vectors into a numpy matrix.
        Returns None if no chunks have valid vectors.
        Shape: (n_valid_chunks, embedding_dim)
        """
        if not chunks:
            return None
        valid = [c for c in chunks if c.get("vector")]
        if not valid:
            return None
        return np.array([c["vector"] for c in valid], dtype=np.float32)

    @staticmethod
    def cosine_similarity_matrix(
        query_vec: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query_vec and every row of matrix.
        Delegates to module-level function for reuse.
        """
        return cosine_similarity_matrix(query_vec, matrix)
