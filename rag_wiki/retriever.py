"""
RagWikiRetriever — the main entry point.

Orchestrates the three-tier retrieval flow:
  1. PINNED docs  → always injected into context
  2. CLAIMED docs → direct local cache hit (skips vector search)
  3. Global RAG   → fallback vector similarity search

Wraps any LangChain BaseRetriever as the global RAG backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)


@dataclass
class RagWikiRetrieverConfig:
    fetch_threshold:     int         = 3
    reset_threshold:     int         = 3     # queries without a hit before fetch_count resets
    no_resiluggest_days: int         = 30    # kept for API compat
    wiki_save_dir:       Optional[str] = None  # directory to save accepted docs; None = disabled
    decay:               DecayConfig  = None

    def __post_init__(self):
        if self.decay is None:
            self.decay = DecayConfig()


class RagWikiRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that adds a personal knowledge cache
    on top of any existing retriever.

    Usage:
        retriever = RagWikiRetriever(
            user_id          = "user-123",
            global_retriever = your_existing_retriever,
            state_store      = MemoryStateStore(),
        )
        docs = retriever.invoke("your query")
        print(retriever.last_provenance.render())
    """

    user_id:          str
    global_retriever: Any
    state_store:      Any
    config:           Any
    last_provenance:  Optional[Any] = None
    on_suggestion:    Optional[Any] = None

    # Internal components — set via object.__setattr__ after Pydantic init
    _sm:      Any = None
    _counter: Any = None
    _decay:   Any = None
    _builder: Any = None

    # Cache: doc_id → page_content from last global retrieval
    _content_cache: Any = None

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

        sm = StateMachine()
        object.__setattr__(self, "_sm", sm)
        object.__setattr__(self, "_counter", FetchCounter(
            store               = store,
            state_machine       = sm,
            fetch_threshold     = cfg.fetch_threshold,
            no_resiluggest_days = cfg.no_resiluggest_days,
            reset_threshold     = cfg.reset_threshold,
        ))
        object.__setattr__(self, "_decay", DecayEngine(
            store         = store,
            state_machine = sm,
            config        = cfg.decay,
        ))
        object.__setattr__(self, "_builder", ProvenanceBuilder())
        object.__setattr__(self, "_content_cache", {})

    # ─── LangChain interface ───────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        retrieved_meta: list[dict]            = []
        user_records:   dict[str, UserDocRecord] = {}
        suggestion:     Optional[SuggestionEvent] = None
        docs:           list[Document]        = []

        # ── Step 1: Inject PINNED documents ───────────────────────────────────
        pinned = self.state_store.list_pinned(self.user_id)
        for record in pinned:
            if record.full_content:
                docs.append(Document(
                    page_content = record.full_content,
                    metadata     = {
                        "doc_id":     record.doc_id,
                        "doc_title":  record.doc_title,
                        "doc_path":   record.doc_path,
                        "from_cache": True,
                        "user_state": DocumentState.PINNED.value,
                    }
                ))
                user_records[record.doc_id] = record
                retrieved_meta.append({
                    "doc_id":     record.doc_id,
                    "doc_title":  record.doc_title,
                    "doc_path":   record.doc_path,
                    "from_cache": True,
                })

        # ── Step 2: Check CLAIMED cache ───────────────────────────────────────
        claimed     = self.state_store.list_claimed(self.user_id)
        claimed_ids = {r.doc_id for r in claimed}

        for record in claimed:
            if record.full_content and self._is_relevant(query, record.full_content):
                docs.append(Document(
                    page_content = record.full_content,
                    metadata     = {
                        "doc_id":     record.doc_id,
                        "doc_title":  record.doc_title,
                        "doc_path":   record.doc_path,
                        "from_cache": True,
                        "user_state": DocumentState.CLAIMED.value,
                    }
                ))
                user_records[record.doc_id] = record
                retrieved_meta.append({
                    "doc_id":     record.doc_id,
                    "doc_title":  record.doc_title,
                    "doc_path":   record.doc_path,
                    "from_cache": True,
                })

        # ── Step 3: Global RAG fallback ───────────────────────────────────────
        global_docs = self.global_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
        )

        # Track which doc_ids we've already incremented this query (avoid double-counting)
        incremented_this_query: set[str] = set()

        for gdoc in global_docs:
            meta = gdoc.metadata or {}

            # ── FIX: resolve doc_id — fall back to source if doc_id missing ──
            doc_id = (
                meta.get("doc_id")
                or meta.get("source")
                or "unknown"
            )
            doc_title = (
                meta.get("doc_title")
                or meta.get("source", "").split("/")[-1]
                or "Unknown Document"
            )
            doc_path = meta.get("doc_path") or meta.get("source") or ""

            # Skip if already served from personal cache
            if doc_id in claimed_ids:
                continue

            # Normalise metadata on the doc so downstream always has doc_id
            gdoc.metadata["doc_id"]    = doc_id
            gdoc.metadata["doc_title"] = doc_title
            gdoc.metadata["doc_path"]  = doc_path

            docs.append(gdoc)

            # Cache content so accept_suggestion can use it without re-fetching
            content_cache = object.__getattribute__(self, "_content_cache")
            if doc_id not in content_cache:
                content_cache[doc_id] = gdoc.page_content

            retrieved_meta.append({
                "doc_id":          doc_id,
                "doc_title":       doc_title,
                "doc_path":        doc_path,
                "chunk_index":     meta.get("chunk_index"),
                "total_chunks":    meta.get("total_chunks"),
                "section_heading": meta.get("section_heading"),
                "from_cache":      False,
            })

            # ── Step 4: Increment fetch counter once per doc per query ────────
            if doc_id not in incremented_this_query:
                incremented_this_query.add(doc_id)
                event = self._counter.record_fetch(
                    user_id   = self.user_id,
                    doc_id    = doc_id,
                    doc_title = doc_title,
                    doc_path  = doc_path,
                )

                # ── Step 5: Fire suggestion if threshold crossed ───────────────
                if event is not None:
                    if self.on_suggestion:
                        self.on_suggestion(event)
                    if suggestion is None:
                        suggestion = event

            # Refresh record for provenance
            record = self.state_store.get(self.user_id, doc_id)
            if record:
                user_records[doc_id] = record

        # ── Step 6: Record misses for docs not retrieved this query ──────────
        # Increment queries_missed for all SURFACED/SUGGESTED docs that didn't
        # appear in this query's results. FetchCounter resets fetch_count when
        # queries_missed reaches reset_threshold.
        retrieved_ids = {m["doc_id"] for m in retrieved_meta if not m.get("from_cache")}
        tracked_docs  = self.state_store.list_surfaced(self.user_id)
        for record in tracked_docs:
            if record.doc_id not in retrieved_ids:
                self._counter.record_miss(self.user_id, record.doc_id)

        # ── Step 7: Build provenance ───────────────────────────────────────────
        provenance = self._builder.build(
            retrieved_docs = retrieved_meta,
            user_records   = user_records,
            suggestion     = suggestion,
        )
        object.__setattr__(self, "last_provenance", provenance)

        return docs

    # ─── User action API ──────────────────────────────────────────────────────

    def accept_suggestion(self, doc_id: str, full_content: Optional[str] = None) -> UserDocRecord:
        """
        User confirmed saving a document to their personal KB.

        Content resolution order:
          1. ``full_content`` argument if provided
          2. Full file read from ``doc_path`` on disk
          3. Cached chunk from last retrieval (fallback)

        If ``config.wiki_save_dir`` is set, a timestamped copy is written there.
        """
        content = full_content

        # Try to read the full file from disk using the stored doc_path
        if not content:
            record = self.state_store.get(self.user_id, doc_id)
            doc_path = record.doc_path if record else ""
            if doc_path and os.path.isfile(doc_path):
                try:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except OSError as exc:
                    logger.warning("Could not read doc_path=%r: %s", doc_path, exc)

        # Fall back to cached chunk if file not readable
        if not content:
            content_cache = object.__getattribute__(self, "_content_cache")
            content = content_cache.get(doc_id, "")
            if content:
                logger.debug("accept_suggestion: using cached chunk for doc_id=%r", doc_id)

        # Optionally save a copy to wiki_save_dir
        if content and self.config.wiki_save_dir:
            try:
                os.makedirs(self.config.wiki_save_dir, exist_ok=True)
                record = self.state_store.get(self.user_id, doc_id)
                doc_title = record.doc_title if record else doc_id.split("/")[-1]
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                safe_title = "".join(
                    c if c.isalnum() or c in " ._-" else "_" for c in doc_title
                )
                wiki_filename = f"{timestamp}_{safe_title}"
                wiki_path = os.path.join(self.config.wiki_save_dir, wiki_filename)
                with open(wiki_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info("Saved doc copy to %s", wiki_path)
            except OSError as exc:
                logger.warning("Could not save doc copy to wiki_save_dir: %s", exc)

        return self._counter.accept_suggestion(self.user_id, doc_id, content)

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
        """Manually trigger decay scoring. Call from a daily cron in production."""
        return self._decay.run_for_user(self.user_id)

    # ─── Private ──────────────────────────────────────────────────────────────

    def _is_relevant(self, query: str, content: str) -> bool:
        """
        Lightweight keyword overlap check for CLAIMED doc cache hits.
        Override with embedding similarity for stricter matching.
        """
        query_terms   = set(query.lower().split())
        content_lower = content.lower()
        matches       = sum(1 for term in query_terms if term in content_lower)
        return matches >= max(1, len(query_terms) // 3)