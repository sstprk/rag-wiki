"""
RagWikiRetriever — the main entry point.

Orchestrates the three-tier retrieval flow:
  1. PINNED + CLAIMED docs → semantic chunk search (or keyword fallback)
  2. Global RAG            → fallback vector similarity search

Wraps any LangChain BaseRetriever as the global RAG backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.storage.memory import MemoryStateStore
from rag_wiki.storage.chunk_store import ChunkStore
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.lifecycle.fetch_counter import FetchCounter, SuggestionEvent
from rag_wiki.lifecycle.decay_engine import DecayEngine, DecayConfig
from rag_wiki.transparency.provenance import ProvenanceBlock, ProvenanceBuilder

logger = logging.getLogger(__name__)


def _resolve_embedding_model(
    explicit_model: Optional[Any],
    global_retriever: Any,
) -> Optional[Any]:
    """
    Resolve the embedding model to use for semantic cache search.

    Priority:
      1. Use explicit_model if provided.
      2. Try to extract from global_retriever via common attribute names.
      3. Try to extract from global_retriever.vectorstore (LangChain VectorStoreRetriever).
      4. Return None — keyword fallback will be used.
    """
    if explicit_model is not None:
        return explicit_model

    for attr in ("embeddings", "embedding_function", "embedding", "_embeddings"):
        model = getattr(global_retriever, attr, None)
        if model is not None and callable(getattr(model, "embed_query", None)):
            return model

    # LangChain VectorStoreRetriever exposes the vectorstore via .vectorstore
    vectorstore = getattr(global_retriever, "vectorstore", None)
    if vectorstore is not None:
        for attr in ("embeddings", "embedding_function", "embedding", "_embeddings"):
            model = getattr(vectorstore, attr, None)
            if model is not None and callable(getattr(model, "embed_query", None)):
                return model

    return None


@dataclass
class RagWikiRetrieverConfig:
    fetch_threshold:      int          = 3
    reset_threshold:      int          = 3       # queries without a hit before fetch_count resets
    no_resiluggest_days:  int          = 30      # kept for API compat
    wiki_save_dir:        Optional[str] = None   # directory to save accepted docs; None = disabled
    similarity_threshold: float        = 0.75   # cosine similarity cutoff for cache hits
    local_top_k:          int          = 3       # max chunks to inject per cached doc
    decay:                DecayConfig  = None

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
    _sm:             Any = None
    _counter:        Any = None
    _decay:          Any = None
    _builder:        Any = None
    _embedding_model: Any = None
    _chunk_store:    Any = None

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
        embedding_model:  Optional[Any] = None,
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
        chunk_store = ChunkStore(wiki_save_dir=cfg.wiki_save_dir)
        resolved_model = _resolve_embedding_model(embedding_model, global_retriever)

        object.__setattr__(self, "_sm", sm)
        object.__setattr__(self, "_embedding_model", resolved_model)
        object.__setattr__(self, "_chunk_store", chunk_store)
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
            chunk_store   = chunk_store,
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
        retrieved_meta: list[dict]               = []
        user_records:   dict[str, UserDocRecord] = {}
        suggestion:     Optional[SuggestionEvent] = None
        docs:           list[Document]           = []

        # ── Embed query once — reused for all cache lookups ───────────────────
        query_vec   = self._embed_query(query)
        chunk_store = object.__getattribute__(self, "_chunk_store")

        # ── Step 1+2: Unified semantic search over CLAIMED + PINNED cache ─────
        cached_records = (
            self.state_store.list_pinned(self.user_id) +
            self.state_store.list_claimed(self.user_id)
        )
        served_doc_ids = set()

        for record in cached_records:
            chunks = chunk_store.load_chunks(self.user_id, record.doc_id)
            now    = datetime.now(timezone.utc)

            if query_vec is not None and chunks:
                # ── Semantic path ──────────────────────────────────────────────
                matrix = ChunkStore.build_matrix(chunks)
                if matrix is not None:
                    scores   = ChunkStore.cosine_similarity_matrix(query_vec, matrix)
                    hit_mask = scores >= self.config.similarity_threshold

                    if hit_mask.any():
                        # Sort hits by score desc, take top local_top_k
                        hit_indices = sorted(
                            [i for i, h in enumerate(hit_mask) if h],
                            key=lambda i: scores[i],
                            reverse=True,
                        )[:self.config.local_top_k]

                        # Record hits for decay tracking
                        real_hit_indices = [chunks[i]["chunk_index"] for i in hit_indices]
                        chunk_store.record_hits(
                            self.user_id, record.doc_id, real_hit_indices, now
                        )

                        # Reset miss streak
                        if record.cache_miss_streak != 0:
                            record.cache_miss_streak = 0
                            self.state_store.upsert(record)

                        # Stamp last_fetched_at so decay recency signal stays alive
                        record.last_fetched_at = now
                        self.state_store.upsert(record)

                        hit_texts = "\n\n".join(chunks[i]["text"] for i in hit_indices)
                        docs.append(Document(
                            page_content = hit_texts,
                            metadata     = {
                                "doc_id":      record.doc_id,
                                "doc_title":   record.doc_title,
                                "doc_path":    record.doc_path,
                                "from_cache":  True,
                                "user_state":  record.user_state.value,
                                "chunks_used": hit_indices,
                                "scores":      [float(scores[i]) for i in hit_indices],
                            }
                        ))
                        served_doc_ids.add(record.doc_id)
                        user_records[record.doc_id] = record
                        retrieved_meta.append({
                            "doc_id":      record.doc_id,
                            "doc_title":   record.doc_title,
                            "doc_path":    record.doc_path,
                            "from_cache":  True,
                            "chunks_used": hit_indices,
                        })

                    else:
                        # No chunks hit — increment miss streak
                        record.cache_miss_streak += 1

                        if record.cache_miss_streak >= self.config.decay.max_cache_miss_streak:
                            logger.info(
                                "Auto-demoting doc_id=%r for user=%r "
                                "(cache_miss_streak=%d)",
                                record.doc_id, self.user_id, record.cache_miss_streak,
                            )
                            try:
                                record = self._sm.transition(
                                    record, DocumentState.DEMOTED, now=now,
                                )
                            except Exception:
                                pass
                            chunk_store.delete(self.user_id, record.doc_id)

                        self.state_store.upsert(record)

            else:
                # ── Keyword fallback (no embedding model or no chunks yet) ─────
                if record.full_content and self._is_relevant(query, record.full_content):
                    docs.append(Document(
                        page_content = record.full_content,
                        metadata     = {
                            "doc_id":     record.doc_id,
                            "doc_title":  record.doc_title,
                            "doc_path":   record.doc_path,
                            "from_cache": True,
                            "user_state": record.user_state.value,
                        }
                    ))
                    served_doc_ids.add(record.doc_id)
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

        # Track which doc_ids we've already incremented this query
        incremented_this_query: set[str] = set()

        for gdoc in global_docs:
            meta = gdoc.metadata or {}

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
            if doc_id in served_doc_ids:
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
        retrieved_ids = {m["doc_id"] for m in retrieved_meta if not m.get("from_cache")}
        tracked_docs  = self.state_store.list_surfaced(self.user_id)
        for record in tracked_docs:
            if record.doc_id not in retrieved_ids:
                self._counter.record_miss(self.user_id, record.doc_id)

        # ── Step 7: Build provenance ──────────────────────────────────────────
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
        Chunk accumulation already happened at retrieval time via _accumulate_chunks.
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

        # Embed and store all chunks since the document is now approved
        if content:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                doc_to_split = Document(page_content=content, metadata={"doc_id": doc_id})
                split_docs = splitter.split_documents([doc_to_split])

                for i, doc in enumerate(split_docs):
                    doc.metadata["chunk_index"] = i
                
                self._accumulate_chunks_batch([(doc_id, doc) for doc in split_docs])
                logger.info("Chunked, embedded and saved %d chunks for doc_id=%r", len(split_docs), doc_id)
            except Exception as exc:
                logger.warning("Failed to chunk and embed doc_id=%r: %s", doc_id, exc)

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

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed the query once. Returns None if no embedding model resolved."""
        model = object.__getattribute__(self, "_embedding_model")
        if model is None:
            return None
        vec = model.embed_query(query)
        return np.array(vec, dtype=np.float32)

    def _accumulate_chunks_batch(
        self, docs_to_embed: list[tuple[str, Document]]
    ) -> None:
        """
        Embed all chunks from a single query in one batch call to
        embed_documents(), then store them. This avoids N separate API
        calls when multiple chunks come back from global RAG in one query.
        Falls back to embed_query per-doc if embed_documents is unavailable.
        """
        model       = object.__getattribute__(self, "_embedding_model")
        chunk_store = object.__getattribute__(self, "_chunk_store")
        if model is None or not docs_to_embed:
            return

        texts = [gdoc.page_content for _, gdoc in docs_to_embed]

        try:
            if callable(getattr(model, "embed_documents", None)):
                vectors = model.embed_documents(texts)
            else:
                # Fallback: embed one at a time if embed_documents not available
                vectors = [model.embed_query(t) for t in texts]
        except Exception:
            return  # embedding failure is non-fatal

        import zlib
        for (doc_id, gdoc), vector in zip(docs_to_embed, vectors):
            meta = gdoc.metadata or {}
            chunk_idx = meta.get("chunk_index")
            if chunk_idx is None:
                chunk_idx = zlib.crc32(gdoc.page_content.encode("utf-8"))

            chunk = {
                "chunk_index": chunk_idx,
                "text":        gdoc.page_content,
                "vector":      vector,
                "section":     meta.get("section_heading", ""),
                "hit_count":   0,
                "last_hit_at": None,
            }
            chunk_store.add_chunks(self.user_id, doc_id, [chunk])



    def _is_relevant(self, query: str, content: str) -> bool:
        """
        Lightweight keyword overlap check for cache hits when no embedding
        model is available.
        """
        query_terms   = set(query.lower().split())
        content_lower = content.lower()
        matches       = sum(1 for term in query_terms if term in content_lower)
        return matches >= max(1, len(query_terms) // 3)
