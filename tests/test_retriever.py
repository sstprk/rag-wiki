"""
Integration tests for RagWikiRetriever.
Uses a MockRetriever instead of a real vector DB.
"""

import pytest
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig, DocumentState
from rag_wiki.storage.sqlite import SQLiteStateStore
from rag_wiki.lifecycle.fetch_counter import AutoSaveEvent


# ─── Mock helpers ─────────────────────────────────────────────────────────────

class MockEmbeddingModel:
    """
    Returns [1.0, 0.0] for text containing "relevant".
    Returns [0.0, 1.0] for all other text.
    Cosine similarity between relevant/relevant = 1.0
    Cosine similarity between relevant/irrelevant = 0.0
    """
    def embed_query(self, text: str) -> list:
        if "relevant" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_documents(self, texts: list) -> list:
        return [self.embed_query(t) for t in texts]


class MockGlobalRetriever(BaseRetriever):
    """Returns a fixed set of documents with proper rag_wiki metadata."""

    docs: List[Document] = []

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


def make_global_doc(doc_id: str, title: str, content: str, chunk_index: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id":       doc_id,
            "doc_title":    title,
            "doc_path":     f"/kb/{doc_id}.pdf",
            "chunk_index":  chunk_index,
            "total_chunks": 10,
        }
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    return SQLiteStateStore("sqlite:///:memory:")

@pytest.fixture
def global_retriever():
    return MockGlobalRetriever(docs=[
        make_global_doc("doc-1", "Market Report", "This is the relevant market report content.", 0),
        make_global_doc("doc-2", "Legal Brief",   "This is the legal brief content.", 1),
    ])

@pytest.fixture
def retriever(store, global_retriever):
    return RagWikiRetriever(
        user_id          = "user-test",
        global_retriever = global_retriever,
        state_store      = store,
        config           = RagWikiRetrieverConfig(fetch_threshold=3),
    )

@pytest.fixture
def retriever_with_embeddings(store, global_retriever):
    return RagWikiRetriever(
        user_id          = "user-test",
        global_retriever = global_retriever,
        state_store      = store,
        config           = RagWikiRetrieverConfig(
            fetch_threshold      = 3,
            similarity_threshold = 0.5,
            local_top_k          = 3,
        ),
        embedding_model  = MockEmbeddingModel(),
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestBasicRetrieval:
    def test_returns_global_docs_on_first_query(self, retriever):
        docs = retriever.invoke("market report")
        assert len(docs) == 2

    def test_provenance_is_set_after_query(self, retriever):
        retriever.invoke("anything")
        assert retriever.last_provenance is not None
        assert len(retriever.last_provenance.sources) == 2

    def test_provenance_render_returns_string(self, retriever):
        retriever.invoke("anything")
        rendered = retriever.last_provenance.render()
        assert isinstance(rendered, str)
        assert "Sources used" in rendered

    def test_fetch_count_increments(self, retriever, store):
        retriever.invoke("market")
        record = store.get("user-test", "doc-1")
        assert record is not None
        assert record.fetch_count == 1


class TestAutoSaveFlow:
    def test_auto_save_fires_at_threshold(self, retriever):
        auto_save_events = []
        retriever.on_auto_save = auto_save_events.append

        for _ in range(3):
            retriever.invoke("market report")

        assert len(auto_save_events) >= 1
        assert auto_save_events[0].doc_id in ("doc-1", "doc-2")

    def test_no_auto_save_before_threshold(self, retriever):
        events = []
        retriever.on_auto_save = events.append

        for _ in range(2):
            retriever.invoke("market report")

        assert len(events) == 0

    def test_auto_save_transitions_to_claimed(self, retriever, store):
        """Auto-save automatically transitions to CLAIMED without user action."""
        for _ in range(3):
            retriever.invoke("market report")

        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.CLAIMED

    def test_auto_save_stores_content(self, retriever, store):
        """Auto-save stores full_content on the record."""
        for _ in range(3):
            retriever.invoke("market report")

        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.CLAIMED
        # Content should be stored (from cache or doc_path)
        assert record.full_content is not None
        assert len(record.full_content) > 0


class TestCacheRetrieval:
    def test_auto_saved_doc_served_from_cache(self, retriever_with_embeddings, store):
        """Auto-saved doc with relevant content is served from cache on relevant query."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")

        # After 3 queries doc-1 should be auto-saved (CLAIMED)
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.CLAIMED

        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1

    def test_provenance_shows_from_cache_true(self, retriever_with_embeddings):
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")

        retriever_with_embeddings.invoke("relevant query")
        sources = {s.doc_id: s for s in retriever_with_embeddings.last_provenance.sources}
        assert sources["doc-1"].from_cache is True


class TestUserSignals:
    def test_thumbs_up_increases_explicit_signal(self, retriever, store):
        retriever.invoke("anything")
        retriever.thumbs_up("doc-1")
        record = store.get("user-test", "doc-1")
        assert record.explicit_signal > 0.0

    def test_force_remove_demotes_claimed_doc(self, retriever, store):
        for _ in range(3):
            retriever.invoke("market")
        # doc-1 is now auto-saved as CLAIMED
        retriever.force_remove("doc-1")
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.DEMOTED

    def test_force_pin_pins_claimed_doc(self, retriever, store):
        for _ in range(3):
            retriever.invoke("market")
        # doc-1 is now auto-saved as CLAIMED
        retriever.force_pin("doc-1")
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.PINNED


# ─── Embedding model resolution ───────────────────────────────────────────────

class TestEmbeddingModelResolution:
    def test_resolved_from_retriever_attribute(self, store):
        """Auto-resolve embedding model from global_retriever.embeddings."""
        class RetrieverWithEmbeddings(BaseRetriever):
            embeddings: object = None
            model_config = {"arbitrary_types_allowed": True}
            def _get_relevant_documents(self, query, *, run_manager):
                return []

        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = RetrieverWithEmbeddings(embeddings=MockEmbeddingModel()),
            state_store      = store,
            embedding_model  = None,
        )
        assert object.__getattribute__(r, "_embedding_model") is not None

    def test_resolved_from_vectorstore_attribute(self, store):
        """Auto-resolve embedding model from global_retriever.vectorstore.embedding_function."""
        class FakeVectorStore:
            embedding_function = MockEmbeddingModel()

        class RetrieverWithVectorStore(BaseRetriever):
            vectorstore: object = None
            model_config = {"arbitrary_types_allowed": True}
            def _get_relevant_documents(self, query, *, run_manager):
                return []

        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = RetrieverWithVectorStore(vectorstore=FakeVectorStore()),
            state_store      = store,
            embedding_model  = None,
        )
        assert object.__getattribute__(r, "_embedding_model") is not None

    def test_no_embedding_model_uses_keyword_fallback(self, store):
        """No embedding model → keyword fallback, no crash."""
        retriever_no_embed = MockGlobalRetriever(docs=[])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = retriever_no_embed,
            state_store      = store,
            embedding_model  = None,
        )
        # Manually claim a doc with matching keyword content
        from rag_wiki.storage.base import UserDocRecord, DocumentState
        record = UserDocRecord(
            user_id="u", doc_id="d1", doc_title="T", doc_path="/p",
            user_state=DocumentState.CLAIMED, full_content="market report content",
        )
        store.upsert(record)
        docs = r.invoke("market report")
        cached = [d for d in docs if d.metadata.get("from_cache")]
        assert len(cached) == 1


# ─── Semantic cache tests ──────────────────────────────────────────────────────

class TestSemanticCache:
    def test_semantic_hit_injects_matching_chunks(self, retriever_with_embeddings, store):
        """Auto-saved doc with relevant content is returned on relevant query."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        # doc-1 is now auto-saved

        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1
        assert cached[0].metadata["from_cache"] is True
        assert "chunks_used" in cached[0].metadata
        assert len(cached[0].metadata["chunks_used"]) > 0

    def test_semantic_miss_does_not_inject(self, store):
        """Auto-saved doc with irrelevant content is NOT returned on relevant query."""
        irrelevant_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-x", "Irrelevant", "completely unrelated xyz content", 0),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = irrelevant_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(
                fetch_threshold      = 1,
                similarity_threshold = 0.5,
            ),
            embedding_model  = MockEmbeddingModel(),
        )
        r.invoke("irrelevant query")
        # doc-x is auto-saved after 1 fetch
        record = store.get("u", "doc-x")
        assert record.user_state == DocumentState.CLAIMED

        docs = r.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-x"]
        assert len(cached) == 0

    def test_cache_miss_streak_increments(self, store):
        """Miss streak increments when no chunk scores above threshold."""
        irrelevant_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-x", "Irrelevant", "completely unrelated xyz content", 0),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = irrelevant_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(
                fetch_threshold      = 1,
                similarity_threshold = 0.5,
            ),
            embedding_model  = MockEmbeddingModel(),
        )
        r.invoke("irrelevant query")
        # doc-x is auto-saved after 1 fetch
        r.invoke("relevant query")  # miss
        record = store.get("u", "doc-x")
        assert record.cache_miss_streak == 1

    def test_cache_miss_streak_resets_on_hit(self, retriever_with_embeddings, store):
        """Miss streak resets to 0 when a chunk matches."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        # doc-1 is now auto-saved

        # Manually set a non-zero streak
        record = store.get("user-test", "doc-1")
        record.cache_miss_streak = 5
        store.upsert(record)
        # Query with matching content — should reset streak
        retriever_with_embeddings.invoke("relevant query")
        record = store.get("user-test", "doc-1")
        assert record.cache_miss_streak == 0

    def test_auto_demotion_on_streak(self, store):
        """Doc is demoted after max_cache_miss_streak consecutive misses."""
        from rag_wiki.lifecycle.decay_engine import DecayConfig
        irrelevant_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-x", "Irrelevant", "completely unrelated xyz content", 0),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = irrelevant_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(
                fetch_threshold      = 1,
                similarity_threshold = 0.5,
                decay                = DecayConfig(max_cache_miss_streak=2),
            ),
            embedding_model  = MockEmbeddingModel(),
        )
        r.invoke("irrelevant query")
        # doc-x is auto-saved after 1 fetch
        # 3 misses — should trigger demotion after 2nd miss
        for _ in range(3):
            r.invoke("relevant query")
        record = store.get("u", "doc-x")
        assert record.user_state == DocumentState.DEMOTED

    def test_chunks_accumulate_during_retrieval(self, retriever_with_embeddings):
        """Chunks are stored at retrieval time."""
        retriever_with_embeddings.invoke("relevant query")
        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        assert len(chunks) >= 1
        assert chunks[0]["vector"] is not None
        assert len(chunks[0]["vector"]) > 0

    def test_last_fetched_at_stamped_on_cache_hit(self, retriever_with_embeddings, store):
        """last_fetched_at is updated when a cache hit occurs."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        # doc-1 is now auto-saved

        retriever_with_embeddings.invoke("relevant query")
        record = store.get("user-test", "doc-1")
        assert record.last_fetched_at is not None


# ─── Incremental chunk accumulation tests ──────────────────────────────────────

class TestIncrementalChunkAccumulation:
    def test_chunks_accumulate_during_retrieval(self, retriever_with_embeddings):
        """Chunks are stored at retrieval time, before auto-save."""
        retriever_with_embeddings.invoke("relevant query")
        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        assert len(chunks) >= 1
        assert chunks[0]["vector"] is not None
        assert len(chunks[0]["vector"]) > 0

    def test_batch_embed_called_once_per_query(self, store):
        """embed_documents is called exactly once per query for all new chunks."""
        call_log = {"count": 0, "texts": []}

        class CountingEmbeddingModel(MockEmbeddingModel):
            def embed_documents(self, texts):
                call_log["count"] += 1
                call_log["texts"] = list(texts)
                return super().embed_documents(texts)

        multi_doc_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-a", "Doc A", "relevant content A", chunk_index=0),
            make_global_doc("doc-a", "Doc A", "relevant content A2", chunk_index=1),
            make_global_doc("doc-b", "Doc B", "irrelevant content B", chunk_index=0),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = multi_doc_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(fetch_threshold=5),
            embedding_model  = CountingEmbeddingModel(),
        )
        r.invoke("relevant query")

        assert call_log["count"] == 1
        assert len(call_log["texts"]) == 3

    def test_accumulation_deduplicates_across_queries(self, retriever_with_embeddings):
        """Repeated queries for the same chunks don't create duplicates."""
        retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.invoke("relevant query")

        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        indices = [c["chunk_index"] for c in chunks]
        # No duplicate chunk_index values
        assert len(indices) == len(set(indices))


# ─── always_full_doc tests ───────────────────────────────────────────────────

class TestAlwaysFullDoc:
    def _auto_save_doc(self, retriever, doc_id):
        """Helper: invoke enough times to cross auto-save threshold."""
        # fetch_threshold=3 in retriever_with_embeddings fixture
        for _ in range(3):
            retriever.invoke("relevant query")

    def test_always_full_doc_injects_full_content_on_hit(
        self, retriever_with_embeddings, store
    ):
        """When always_full_doc=True, full_content is injected on a cache hit."""
        self._auto_save_doc(retriever_with_embeddings, "doc-1")
        # doc-1 is now CLAIMED via auto-save

        # Update full_content to a known value for testing
        record = store.get("user-test", "doc-1")
        full_text = "full document text with relevant content"
        record.full_content = full_text
        store.upsert(record)

        retriever_with_embeddings.set_always_full_doc("doc-1", enabled=True)

        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1
        assert cached[0].page_content == full_text
        assert cached[0].metadata["full_doc_injected"] is True

    def test_always_full_doc_false_injects_chunks_only(
        self, retriever_with_embeddings, store
    ):
        """When always_full_doc=False (default), only matched chunks are injected."""
        self._auto_save_doc(retriever_with_embeddings, "doc-1")
        full_text = "full document text with relevant content"
        record = store.get("user-test", "doc-1")
        record.full_content = full_text
        store.upsert(record)
        # Do NOT call set_always_full_doc — default is False

        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1
        assert cached[0].page_content != full_text  # only chunks, not full doc
        assert cached[0].metadata["full_doc_injected"] is False

    def test_always_full_doc_falls_back_to_chunks_when_content_unavailable(
        self, retriever_with_embeddings, store
    ):
        """Falls back to chunks gracefully when full content is unavailable."""
        self._auto_save_doc(retriever_with_embeddings, "doc-1")
        retriever_with_embeddings.set_always_full_doc("doc-1", enabled=True)

        # Wipe full_content and set a bad path
        record = store.get("user-test", "doc-1")
        record.full_content = None
        record.doc_path = "/nonexistent/path/doc.txt"
        store.upsert(record)

        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1  # no crash
        assert cached[0].metadata["full_doc_injected"] is False

    def test_set_always_full_doc_raises_on_unclaimed_doc(
        self, retriever_with_embeddings
    ):
        """Raises ValueError when doc doesn't exist in the store."""
        with pytest.raises(ValueError, match="No cached record"):
            retriever_with_embeddings.set_always_full_doc("nonexistent-id")

    def test_set_always_full_doc_raises_on_surfaced_doc(
        self, retriever_with_embeddings, store
    ):
        """Raises ValueError when doc is SURFACED (not yet auto-saved)."""
        retriever_with_embeddings.invoke("relevant query")  # surfaces doc-1
        with pytest.raises(ValueError, match="CLAIMED or PINNED"):
            retriever_with_embeddings.set_always_full_doc("doc-1")

    def test_always_full_doc_decay_still_tracks_normally(
        self, retriever_with_embeddings, store
    ):
        """Decay tracking (hit_count) still updates when always_full_doc=True."""
        self._auto_save_doc(retriever_with_embeddings, "doc-1")
        full_text = "full document text with relevant content"
        record = store.get("user-test", "doc-1")
        record.full_content = full_text
        store.upsert(record)
        retriever_with_embeddings.set_always_full_doc("doc-1", enabled=True)

        retriever_with_embeddings.invoke("relevant query")

        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        total_hits = sum(c["hit_count"] for c in chunks)
        assert total_hits > 0


# ─── Record cache tests (Step 1) ──────────────────────────────────────────────

class TestRecordCache:
    def test_cached_records_avoid_repeated_store_calls(self, store):
        """After initial fetch, list_pinned/list_claimed are cached for TTL."""
        call_counts = {"list_pinned": 0, "list_claimed": 0}
        orig_pinned = store.list_pinned
        orig_claimed = store.list_claimed

        def counting_list_pinned(user_id):
            call_counts["list_pinned"] += 1
            return orig_pinned(user_id)

        def counting_list_claimed(user_id):
            call_counts["list_claimed"] += 1
            return orig_claimed(user_id)

        store.list_pinned = counting_list_pinned
        store.list_claimed = counting_list_claimed

        r = RagWikiRetriever(
            user_id="u", global_retriever=MockGlobalRetriever(docs=[]),
            state_store=store,
            config=RagWikiRetrieverConfig(record_cache_ttl_seconds=30),
        )
        # 3 queries with no writes — list_pinned called once (cached for 2+3)
        r.invoke("q1")
        r.invoke("q2")
        r.invoke("q3")
        assert call_counts["list_pinned"] == 1
        assert call_counts["list_claimed"] == 1

    def test_cache_invalidated_on_upsert_mid_query(self, store):
        """After an auto-save upsert fires, next query re-fetches from store."""
        call_counts = {"list_pinned": 0}
        orig_pinned = store.list_pinned

        def counting_list_pinned(user_id):
            call_counts["list_pinned"] += 1
            return orig_pinned(user_id)

        store.list_pinned = counting_list_pinned

        global_ret = MockGlobalRetriever(docs=[
            make_global_doc("doc-a", "A", "relevant content A"),
        ])
        r = RagWikiRetriever(
            user_id="u", global_retriever=global_ret,
            state_store=store,
            config=RagWikiRetrieverConfig(fetch_threshold=2, record_cache_ttl_seconds=30),
            embedding_model=MockEmbeddingModel(),
        )
        r.invoke("relevant q1")  # fetch 1 — cache primed
        r.invoke("relevant q2")  # fetch 2 — auto-save fires, cache invalidated
        # Next query must re-fetch because upsert fired
        r.invoke("relevant q3")
        assert call_counts["list_pinned"] >= 2

    def test_cache_refetches_after_ttl_expires(self, store):
        """After TTL expires, records are re-fetched from store."""
        from unittest.mock import patch
        from datetime import timedelta

        call_counts = {"list_pinned": 0}
        orig_pinned = store.list_pinned

        def counting_list_pinned(user_id):
            call_counts["list_pinned"] += 1
            return orig_pinned(user_id)

        store.list_pinned = counting_list_pinned

        r = RagWikiRetriever(
            user_id="u", global_retriever=MockGlobalRetriever(docs=[]),
            state_store=store,
            config=RagWikiRetrieverConfig(record_cache_ttl_seconds=1),
        )
        r.invoke("q1")
        assert call_counts["list_pinned"] == 1

        # Manually expire the cache by adjusting cached_at
        rc = object.__getattribute__(r, "_record_cache")
        rc["cached_at"] = rc["cached_at"] - timedelta(seconds=5)

        r.invoke("q2")
        assert call_counts["list_pinned"] == 2


# ─── Matrix cache tests (Step 2) ──────────────────────────────────────────────

class TestMatrixCache:
    def test_matrix_cache_avoids_repeated_chunk_loads(self, retriever_with_embeddings):
        """load_chunks is only called once per doc until chunks change."""
        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        call_count = {"load_chunks": 0}
        orig_load = chunk_store.load_chunks

        def counting_load_chunks(user_id, doc_id):
            if doc_id == "doc-1":
                call_count["load_chunks"] += 1
            return orig_load(user_id, doc_id)

        chunk_store.load_chunks = counting_load_chunks

        # query 1: global fetch (misses cache) -> docs are SURFACED
        retriever_with_embeddings.invoke("relevant query")
        # query 2: global fetch -> docs are SUGGESTED
        retriever_with_embeddings.invoke("relevant query")
        # query 3: global fetch -> docs are CLAIMED (auto-saved)
        retriever_with_embeddings.invoke("relevant query")
        
        # Now doc-1 is CLAIMED, so the cache block runs
        # We need a fresh query to hit the semantic cache
        call_count["load_chunks"] = 0
        
        retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.invoke("relevant query")

        # It should only be called once per doc, so 1 total for doc-1
        assert call_count["load_chunks"] == 1

    def test_cache_refetches_after_accumulate_chunks(self, retriever_with_embeddings):
        """After _accumulate_chunks_batch fires, matrix cache is invalidated."""
        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        call_count = {"load_chunks": 0}
        orig_load = chunk_store.load_chunks

        def counting_load_chunks(user_id, doc_id):
            if doc_id == "doc-1":
                call_count["load_chunks"] += 1
            return orig_load(user_id, doc_id)

        chunk_store.load_chunks = counting_load_chunks

        # Get doc auto-saved
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
            
        call_count["load_chunks"] = 0
        retriever_with_embeddings.invoke("relevant query")
        assert call_count["load_chunks"] == 1
        
        # Manually trigger chunk accumulation which invalidates cache for doc-1
        retriever_with_embeddings._accumulate_chunks_batch([
            ("doc-1", make_global_doc("doc-1", "T", "more content", chunk_index=99))
        ])
        
        # Next query should re-load doc-1 (so +1 call = 2 total)
        retriever_with_embeddings.invoke("relevant query")
        assert call_count["load_chunks"] == 2

    def test_cache_entry_invalidated_on_demotion(self, store):
        """Auto-demotion (chunk_store.delete()) clears matrix cache entry."""
        from rag_wiki.lifecycle.decay_engine import DecayConfig
        irrelevant_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-x", "Irrelevant", "completely unrelated xyz content", 0),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = irrelevant_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(
                fetch_threshold      = 1,
                similarity_threshold = 0.5,
                decay                = DecayConfig(max_cache_miss_streak=1),
            ),
            embedding_model  = MockEmbeddingModel(),
        )
        
        r.invoke("irrelevant query") # doc-x is CLAIMED
        
        # This query will miss the semantic cache for doc-x, causing demotion
        # since max_cache_miss_streak = 1
        r.invoke("relevant query")
        
        # Verify matrix cache is marked dirty
        mc = object.__getattribute__(r, "_matrix_cache")
        assert mc[("u", "doc-x")]["dirty"] is True


# ─── Lazy miss tracking tests (Step 3) ────────────────────────────────────────

class TestLazyMissTracking:
    def test_upsert_deferred_until_threshold(self, store):
        """miss tracking only hits the DB when reset_threshold is reached."""
        irrelevant_retriever = MockGlobalRetriever(docs=[
            make_global_doc("doc-1", "D1", "content"),
        ])
        r = RagWikiRetriever(
            user_id          = "u",
            global_retriever = irrelevant_retriever,
            state_store      = store,
            config           = RagWikiRetrieverConfig(reset_threshold=3),
            embedding_model  = MockEmbeddingModel(),
        )

        # Query 1: doc-1 is fetched and SURFACED
        r.invoke("q1")
        
        # Now doc-1 is in the system. Let's spy on upsert.
        call_count = {"upsert": 0}
        orig_upsert = store.upsert
        def counting_upsert(record):
            if record.doc_id == "doc-1":
                call_count["upsert"] += 1
            orig_upsert(record)
        store.upsert = counting_upsert

        # Swap retriever to return nothing so doc-1 is missed without dirtying cache
        r.global_retriever = MockGlobalRetriever(docs=[])

        # Miss 1: queries_missed = 1 (in cache). DB upsert deferred.
        r.invoke("q2")
        assert call_count["upsert"] == 0

        # Miss 2: queries_missed = 2 (in cache). DB upsert deferred.
        r.invoke("q3")
        assert call_count["upsert"] == 0

        # Miss 3: queries_missed = 3. Hits reset_threshold! Upsert should fire.
        r.invoke("q4")
        assert call_count["upsert"] >= 1  # May be 2 if FetchCounter also upserts
