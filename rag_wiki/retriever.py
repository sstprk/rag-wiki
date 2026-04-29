"""
RagWikiRetriever — the main entry point.

Orchestrates the three-tier retrieval flow:
  1. PINNED docs  → always injected into context
  2. CLAIMED docs → direct local cache hit (skips vector search)
  3. Global RAG   → fallback vector similarity search

Wraps any LangChain BaseRetriever as the global RAG backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.storage.memory import MemoryStateStore
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.lifecycle.fetch_counter import FetchCounter, SuggestionEvent
from rag_wiki.lifecycle.decay_engine import DecayEngine, DecayConfig
from rag_wiki.transparency.provenance import ProvenanceBlock, ProvenanceBuilder


@dataclass
class RagWikiRetrieverConfig:
    fetch_threshold:     int   = 3
    no_resiluggest_days: int   = 30
    decay:               DecayConfig = None

    def __post_init__(self):
        if self.decay is None:
            self.decay = DecayConfig()


class RagWikiRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that adds a personal knowledge cache
    on top of any existing retriever.

    Usage:
        from rag_wiki import RagWikiRetriever
        from rag_wiki.storage.memory import MemoryStateStore

        retriever = RagWikiRetriever(
            user_id          = "user-123",
            global_retriever = your_existing_retriever,
            state_store      = MemoryStateStore(),
        )

        # Drop-in replacement anywhere a LangChain retriever is expected:
        docs = retriever.get_relevant_documents("your query")

        # Access provenance from the last query:
        print(retriever.last_provenance.render())

        # User actions:
        retriever.accept_suggestion(doc_id="...", full_content="...")
        retriever.decline_suggestion(doc_id="...")
        retriever.thumbs_up(doc_id="...")
        retriever.force_pin(doc_id="...")
        retriever.force_remove(doc_id="...")
    """

    # Pydantic fields (LangChain BaseRetriever is a Pydantic model)
    user_id:          str
    global_retriever: Any   # BaseRetriever — typed as Any to avoid Pydantic issues
    state_store:      Any   # StateStore
    config:           Any   # RagWikiRetrieverConfig

    # Internal — not set by caller
    _sm:       Any = None
    _counter:  Any = None
    _decay:    Any = None
    _builder:  Any = None

    # Last provenance block — available after each query
    last_provenance: Optional[Any] = None

    # Optional callback: called with SuggestionEvent when threshold is crossed
    # signature: (event: SuggestionEvent) -> None
    on_suggestion: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        user_id:          str,
        global_retriever: BaseRetriever,
        state_store:      Optional[StateStore] = None,
        config:           Optional[RagWikiRetrieverConfig] = None,
        on_suggestion:    Optional[Callable[[SuggestionEvent], None]] = None,
        **kwargs,
    ):
        cfg   = config or RagWikiRetrieverConfig()
        store = state_store or MemoryStateStore()

        super().__init__(
            user_id          = user_id,
            global_retriever = global_retriever,
            state_store      = store,
            config           = cfg,
            on_suggestion    = on_suggestion,
            **kwargs,
        )

        # Wire up internals after Pydantic init
        sm      = StateMachine()
        object.__setattr__(self, "_sm",      sm)
        object.__setattr__(self, "_counter", FetchCounter(
            store            = store,
            state_machine    = sm,
            fetch_threshold  = cfg.fetch_threshold,
            no_resiluggest_days = cfg.no_resiluggest_days,
        ))
        object.__setattr__(self, "_decay",   DecayEngine(
            store          = store,
            state_machine  = sm,
            config         = cfg.decay,
        ))
        object.__setattr__(self, "_builder", ProvenanceBuilder())

    # ─── LangChain interface ───────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        retrieved_meta: list[dict]      = []
        user_records:   dict[str, UserDocRecord] = {}
        suggestion:     Optional[SuggestionEvent] = None
        docs:           list[Document] = []

        # ── Step 1: Inject PINNED documents ───────────────────────────────────
        pinned = self.state_store.list_pinned(self.user_id)
        for record in pinned:
            if record.full_content:
                doc = Document(
                    page_content = record.full_content,
                    metadata     = {
                        "doc_id":     record.doc_id,
                        "doc_title":  record.doc_title,
                        "doc_path":   record.doc_path,
                        "from_cache": True,
                        "user_state": DocumentState.PINNED.value,
                    }
                )
                docs.append(doc)
                user_records[record.doc_id] = record
                retrieved_meta.append({
                    "doc_id":    record.doc_id,
                    "doc_title": record.doc_title,
                    "doc_path":  record.doc_path,
                    "from_cache": True,
                })

        # ── Step 2: Check CLAIMED cache ───────────────────────────────────────
        claimed = self.state_store.list_claimed(self.user_id)
        claimed_ids = {r.doc_id for r in claimed}

        for record in claimed:
            if record.full_content and self._is_relevant(query, record.full_content):
                doc = Document(
                    page_content = record.full_content,
                    metadata     = {
                        "doc_id":     record.doc_id,
                        "doc_title":  record.doc_title,
                        "doc_path":   record.doc_path,
                        "from_cache": True,
                        "user_state": DocumentState.CLAIMED.value,
                    }
                )
                docs.append(doc)
                user_records[record.doc_id] = record
                retrieved_meta.append({
                    "doc_id":    record.doc_id,
                    "doc_title": record.doc_title,
                    "doc_path":  record.doc_path,
                    "from_cache": True,
                })

        # ── Step 3: Global RAG fallback ───────────────────────────────────────
        global_docs = self.global_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
        )

        for gdoc in global_docs:
            meta   = gdoc.metadata or {}
            doc_id = meta.get("doc_id")

            # Skip if already served from cache
            if doc_id and doc_id in claimed_ids:
                continue

            docs.append(gdoc)
            retrieved_meta.append({
                "doc_id":          doc_id,
                "doc_title":       meta.get("doc_title", meta.get("source", "Unknown")),
                "doc_path":        meta.get("doc_path", meta.get("source", "")),
                "chunk_index":     meta.get("chunk_index"),
                "total_chunks":    meta.get("total_chunks"),
                "section_heading": meta.get("section_heading"),
                "from_cache":      False,
            })

            # ── Step 4: Increment fetch counters ─────────────────────────────
            if doc_id:
                event = self._counter.record_fetch(
                    user_id   = self.user_id,
                    doc_id    = doc_id,
                    doc_title = meta.get("doc_title", meta.get("source", "Unknown")),
                    doc_path  = meta.get("doc_path", meta.get("source", "")),
                )

                # ── Step 5: Check for suggestion ─────────────────────────────
                if event is not None:
                    suggestion = event
                    if self.on_suggestion:
                        self.on_suggestion(event)

                # Refresh record for provenance
                record = self.state_store.get(self.user_id, doc_id)
                if record:
                    user_records[doc_id] = record

        # ── Step 6: Build and store provenance ────────────────────────────────
        provenance = self._builder.build(
            retrieved_docs = retrieved_meta,
            user_records   = user_records,
            suggestion     = suggestion,
        )
        object.__setattr__(self, "last_provenance", provenance)

        return docs

    # ─── User action API ──────────────────────────────────────────────────────

    def accept_suggestion(self, doc_id: str, full_content: str) -> UserDocRecord:
        """User confirmed saving a document to their personal KB."""
        return self._counter.accept_suggestion(self.user_id, doc_id, full_content)

    def decline_suggestion(self, doc_id: str) -> UserDocRecord:
        """User dismissed the save suggestion."""
        return self._counter.decline_suggestion(self.user_id, doc_id)

    def thumbs_up(self, doc_id: str) -> None:
        """User marked a source as explicitly useful."""
        self._decay.thumbs_up(self.user_id, doc_id)

    def thumbs_down(self, doc_id: str) -> None:
        """User marked a source as not useful."""
        self._decay.thumbs_down(self.user_id, doc_id)

    def force_pin(self, doc_id: str) -> None:
        """User said 'always include this document'."""
        self._decay.force_pin(self.user_id, doc_id)

    def force_remove(self, doc_id: str) -> None:
        """User said 'remove this from my KB entirely'."""
        self._decay.force_remove(self.user_id, doc_id)

    def run_decay(self) -> list:
        """
        Manually trigger decay scoring for this user.
        In production call this from your daily cron job.
        """
        return self._decay.run_for_user(self.user_id)

    # ─── Private ──────────────────────────────────────────────────────────────

    def _is_relevant(self, query: str, content: str) -> bool:
        """
        Lightweight relevance check for CLAIMED documents.
        Default: simple keyword overlap. Override this with embedding similarity
        if you want stricter cache matching.
        """
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        return matches >= max(1, len(query_terms) // 3)