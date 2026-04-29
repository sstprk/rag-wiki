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


# ─── Mock global retriever ────────────────────────────────────────────────────

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
        make_global_doc("doc-1", "Market Report", "This is the market report content.", 0),
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
        assert record.no_resiluggest_until is not None


class TestCacheRetrieval:
    def test_claimed_doc_served_from_cache(self, retriever, store):
        # First get doc surfaced and then claim it
        for _ in range(3):
            retriever.invoke("market report")
        retriever.accept_suggestion("doc-1", full_content="market report full content")

        # Next query should hit the cache
        docs = retriever.invoke("market report")
        cached = [d for d in docs if d.metadata.get("from_cache") and d.metadata.get("doc_id") == "doc-1"]
        assert len(cached) == 1
        assert cached[0].page_content == "market report full content"

    def test_provenance_shows_from_cache_true(self, retriever):
        for _ in range(3):
            retriever.invoke("market")
        retriever.accept_suggestion("doc-1", full_content="market report full content")

        retriever.invoke("market")
        sources = {s.doc_id: s for s in retriever.last_provenance.sources}
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