"""
Tests for ChunkStore — both memory and disk backends.
Covers: add_chunks, load_chunks, record_hits, get_hit_rate,
        build_matrix, cosine_similarity_matrix, delete, thread safety.
"""

import threading
import pytest
import numpy as np

from rag_wiki.storage.chunk_store import ChunkStore


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_store():
    return ChunkStore(wiki_save_dir=None)


@pytest.fixture
def disk_store(tmp_path):
    return ChunkStore(wiki_save_dir=str(tmp_path))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_chunk(index: int, text: str = "text", vector=None) -> dict:
    return {
        "chunk_index": index,
        "text":        text,
        "vector":      vector or [float(index), 0.0],
        "section":     "",
        "hit_count":   0,
        "last_hit_at": None,
    }


# ─── add_chunks / load_chunks ─────────────────────────────────────────────────

class TestAddChunksMemory:
    def test_add_and_load(self, mem_store):
        chunks = [make_chunk(i) for i in range(3)]
        added = mem_store.add_chunks("u1", "d1", chunks)
        assert added == 3
        loaded = mem_store.load_chunks("u1", "d1")
        assert len(loaded) == 3

    def test_all_duplicates_returns_zero(self, mem_store):
        chunks = [make_chunk(i) for i in range(3)]
        mem_store.add_chunks("u1", "d1", chunks)
        added = mem_store.add_chunks("u1", "d1", chunks)
        assert added == 0
        assert len(mem_store.load_chunks("u1", "d1")) == 3

    def test_partial_new_chunks(self, mem_store):
        mem_store.add_chunks("u1", "d1", [make_chunk(0), make_chunk(1)])
        added = mem_store.add_chunks("u1", "d1", [make_chunk(1), make_chunk(2)])
        assert added == 1
        assert len(mem_store.load_chunks("u1", "d1")) == 3


class TestAddChunksDisk:
    def test_add_and_load(self, disk_store):
        chunks = [make_chunk(i) for i in range(3)]
        added = disk_store.add_chunks("u1", "d1", chunks)
        assert added == 3
        loaded = disk_store.load_chunks("u1", "d1")
        assert len(loaded) == 3

    def test_all_duplicates_returns_zero(self, disk_store):
        chunks = [make_chunk(i) for i in range(3)]
        disk_store.add_chunks("u1", "d1", chunks)
        added = disk_store.add_chunks("u1", "d1", chunks)
        assert added == 0
        assert len(disk_store.load_chunks("u1", "d1")) == 3

    def test_partial_new_chunks(self, disk_store):
        disk_store.add_chunks("u1", "d1", [make_chunk(0), make_chunk(1)])
        added = disk_store.add_chunks("u1", "d1", [make_chunk(1), make_chunk(2)])
        assert added == 1
        assert len(disk_store.load_chunks("u1", "d1")) == 3


# ─── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_by_chunk_index(self, mem_store):
        mem_store.add_chunks("u1", "d1", [make_chunk(0), make_chunk(1)])
        added = mem_store.add_chunks("u1", "d1", [make_chunk(1), make_chunk(2)])
        assert added == 1
        loaded = mem_store.load_chunks("u1", "d1")
        indices = {c["chunk_index"] for c in loaded}
        assert indices == {0, 1, 2}


# ─── record_hits ──────────────────────────────────────────────────────────────

class TestRecordHits:
    def _setup(self, store):
        chunks = [make_chunk(i) for i in range(3)]
        store.add_chunks("u1", "d1", chunks)

    def test_hit_count_increments(self, mem_store):
        from datetime import datetime, timezone
        self._setup(mem_store)
        now = datetime.now(timezone.utc)
        mem_store.record_hits("u1", "d1", [0, 2], now)
        loaded = mem_store.load_chunks("u1", "d1")
        by_index = {c["chunk_index"]: c for c in loaded}
        assert by_index[0]["hit_count"] == 1
        assert by_index[1]["hit_count"] == 0
        assert by_index[2]["hit_count"] == 1

    def test_last_hit_at_set(self, mem_store):
        from datetime import datetime, timezone
        self._setup(mem_store)
        now = datetime.now(timezone.utc)
        mem_store.record_hits("u1", "d1", [0], now)
        loaded = mem_store.load_chunks("u1", "d1")
        by_index = {c["chunk_index"]: c for c in loaded}
        assert by_index[0]["last_hit_at"] is not None
        assert by_index[1]["last_hit_at"] is None


# ─── get_hit_rate ─────────────────────────────────────────────────────────────

class TestGetHitRate:
    def test_empty_returns_zero(self, mem_store):
        assert mem_store.get_hit_rate("u1", "unknown_doc") == 0.0

    def test_partial_hit_rate(self, mem_store):
        from datetime import datetime, timezone
        chunks = [make_chunk(i) for i in range(4)]
        mem_store.add_chunks("u1", "d1", chunks)
        now = datetime.now(timezone.utc)
        mem_store.record_hits("u1", "d1", [0, 2], now)
        assert mem_store.get_hit_rate("u1", "d1") == 0.5


# ─── build_matrix ─────────────────────────────────────────────────────────────

class TestBuildMatrix:
    def test_shape(self):
        chunks = [{"chunk_index": i, "vector": [1.0, 2.0, 3.0]} for i in range(5)]
        matrix = ChunkStore.build_matrix(chunks)
        assert matrix is not None
        assert matrix.shape == (5, 3)

    def test_empty_returns_none(self):
        assert ChunkStore.build_matrix([]) is None

    def test_no_vectors_returns_none(self):
        chunks = [{"chunk_index": 0, "vector": []}]
        assert ChunkStore.build_matrix(chunks) is None


# ─── cosine_similarity_matrix ─────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_unit_vectors(self):
        query  = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        scores = ChunkStore.cosine_similarity_matrix(query, matrix)
        assert abs(scores[0] - 1.0)  < 1e-5
        assert abs(scores[1] - 0.0)  < 1e-5
        assert abs(scores[2] - (-1.0)) < 1e-5

    def test_unnormalised_vectors(self):
        """Scores must be in [-1, 1] even with large-magnitude vectors."""
        query  = np.array([3.0, 0.0], dtype=np.float32)
        matrix = np.array([[5.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        scores = ChunkStore.cosine_similarity_matrix(query, matrix)
        assert abs(scores[0] - 1.0) < 1e-5
        assert abs(scores[1] - 0.0) < 1e-5

    def test_zero_query_vector(self):
        query  = np.array([0.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        scores = ChunkStore.cosine_similarity_matrix(query, matrix)
        assert all(s == 0.0 for s in scores)


# ─── delete ───────────────────────────────────────────────────────────────────

class TestDeleteMemory:
    def test_delete_clears_chunks(self, mem_store):
        mem_store.add_chunks("u1", "d1", [make_chunk(0)])
        mem_store.delete("u1", "d1")
        assert mem_store.load_chunks("u1", "d1") == []

    def test_delete_nonexistent_is_noop(self, mem_store):
        mem_store.delete("u1", "unknown")  # should not raise


class TestDeleteDisk:
    def test_delete_clears_chunks(self, disk_store):
        disk_store.add_chunks("u1", "d1", [make_chunk(0)])
        disk_store.delete("u1", "d1")
        assert disk_store.load_chunks("u1", "d1") == []

    def test_file_removed(self, disk_store, tmp_path):
        disk_store.add_chunks("u1", "d1", [make_chunk(0)])
        path = disk_store._chunk_path("u1", "d1")
        assert path.exists()
        disk_store.delete("u1", "d1")
        assert not path.exists()


# ─── Atomic write ─────────────────────────────────────────────────────────────

class TestAtomicWriteDisk:
    def test_no_tmp_file_after_save(self, disk_store, tmp_path):
        disk_store.save_chunks("u1", "d1", [make_chunk(0)])
        path = disk_store._chunk_path("u1", "d1")
        tmp  = path.with_suffix(".jsonl.tmp")
        assert path.exists()
        assert not tmp.exists()


# ─── Thread safety ────────────────────────────────────────────────────────────

class TestThreadSafetyMemory:
    def test_concurrent_add_chunks(self, mem_store):
        errors = []

        def add(index):
            try:
                mem_store.add_chunks("u1", "d1", [make_chunk(index)])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        loaded = mem_store.load_chunks("u1", "d1")
        indices = {c["chunk_index"] for c in loaded}
        assert len(indices) == 10  # no data loss, no duplicates
