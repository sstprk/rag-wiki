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
from rag_wiki.lifecycle.fetch_counter import SuggestionEvent


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


class TestSuggestionFlow:
    def test_suggestion_fires_at_threshold(self, retriever):
        suggestion_events = []
        retriever.on_suggestion = suggestion_events.append

        for _ in range(3):
            retriever.invoke("market report")

        assert len(suggestion_events) >= 1
        assert suggestion_events[0].doc_id in ("doc-1", "doc-2")

    def test_no_suggestion_before_threshold(self, retriever):
        events = []
        retriever.on_suggestion = events.append

        for _ in range(2):
            retriever.invoke("market report")

        assert len(events) == 0

    def test_accept_suggestion_stores_content(self, retriever, store):
        for _ in range(3):
            retriever.invoke("market report")

        retriever.accept_suggestion("doc-1", full_content="Full document text here.")
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.CLAIMED
        assert record.full_content == "Full document text here."

    def test_decline_suggestion_stays_surfaced(self, retriever, store):
        for _ in range(3):
            retriever.invoke("market report")

        retriever.decline_suggestion("doc-1")
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.SURFACED
        # After decline, next suggestion is scheduled at an escalated fetch count
        assert record.next_suggest_at > record.fetch_count


class TestCacheRetrieval:
    def test_claimed_doc_served_from_cache(self, retriever_with_embeddings, store):
        """Claimed doc with relevant content is served from cache on relevant query."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.accept_suggestion(
            "doc-1", full_content="this is relevant content"
        )
        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1

    def test_provenance_shows_from_cache_true(self, retriever_with_embeddings):
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.accept_suggestion(
            "doc-1", full_content="this is relevant content"
        )
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
        retriever.accept_suggestion("doc-1", full_content="content")
        retriever.force_remove("doc-1")
        record = store.get("user-test", "doc-1")
        assert record.user_state == DocumentState.DEMOTED

    def test_force_pin_pins_claimed_doc(self, retriever, store):
        for _ in range(3):
            retriever.invoke("market")
        retriever.accept_suggestion("doc-1", full_content="content")
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
        """Claimed doc with relevant content is returned on relevant query."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.accept_suggestion(
            "doc-1", full_content="this is relevant content"
        )
        docs = retriever_with_embeddings.invoke("relevant query")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1
        assert cached[0].metadata["from_cache"] is True
        assert "chunks_used" in cached[0].metadata
        assert len(cached[0].metadata["chunks_used"]) > 0

    def test_semantic_miss_does_not_inject(self, store):
        """Claimed doc with irrelevant content is NOT returned on relevant query."""
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
        r.accept_suggestion("doc-x", full_content="completely unrelated xyz content")
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
        r.accept_suggestion("doc-x", full_content="completely unrelated xyz content")
        r.invoke("relevant query")  # miss
        record = store.get("u", "doc-x")
        assert record.cache_miss_streak == 1

    def test_cache_miss_streak_resets_on_hit(self, retriever_with_embeddings, store):
        """Miss streak resets to 0 when a chunk matches."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.accept_suggestion(
            "doc-1", full_content="this is relevant content"
        )
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
        r.accept_suggestion("doc-x", full_content="completely unrelated xyz content")
        # 3 misses — should trigger demotion after 2nd miss
        for _ in range(3):
            r.invoke("relevant query")
        record = store.get("u", "doc-x")
        assert record.user_state == DocumentState.DEMOTED

    def test_chunks_accumulate_after_claim(self, retriever_with_embeddings):
        """Chunks are stored ONLY when the user claims, not at retrieval time."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        
        chunk_store = object.__getattribute__(retriever_with_embeddings, "_chunk_store")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        assert len(chunks) == 0  # No chunks before claim
        
        retriever_with_embeddings.accept_suggestion("doc-1", full_content="relevant content")
        chunks = chunk_store.load_chunks("user-test", "doc-1")
        assert len(chunks) >= 1  # Chunks added after claim

    def test_last_fetched_at_stamped_on_cache_hit(self, retriever_with_embeddings, store):
        """last_fetched_at is updated when a cache hit occurs."""
        for _ in range(3):
            retriever_with_embeddings.invoke("relevant query")
        retriever_with_embeddings.accept_suggestion(
            "doc-1", full_content="this is relevant content"
        )
        retriever_with_embeddings.invoke("relevant query")
        record = store.get("user-test", "doc-1")
        assert record.last_fetched_at is not None
