"""
RagWikiRetriever — the main entry point.

Orchestrates the three-tier retrieval flow:
  1. PINNED + CLAIMED docs → semantic chunk search (or keyword fallback)
  2. Global RAG            → fallback vector similarity search

Auto-save lifecycle:
  Documents fetched repeatedly (fetch_count >= fetch_threshold) are
  automatically saved to the user's personal KB — no manual
  accept/decline needed. Decay + cache-miss streak handle removal.

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
from rag_wiki.lifecycle.fetch_counter import FetchCounter, AutoSaveEvent
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
    fetch_threshold:          int          = 3
    reset_threshold:          int          = 3       # queries without a hit before fetch_count resets
    wiki_save_dir:            Optional[str] = None   # directory to save accepted docs; None = disabled
    similarity_threshold:     float        = 0.75   # cosine similarity cutoff for cache hits
    local_top_k:              int          = 3       # max chunks to inject per cached doc
    record_cache_ttl_seconds: int          = 30      # TTL for in-process record cache
    decay:                    DecayConfig  = None

    def __post_init__(self):
        if self.decay is None:
            self.decay = DecayConfig()


class RagWikiRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that adds a personal knowledge cache
    on top of any existing retriever.

    Documents are automatically saved to the user's personal KB when they
    have been fetched ``fetch_threshold`` times (auto-save). No manual
    accept/decline is needed — the save-delete lifecycle is fully automated.

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
    on_auto_save:     Optional[Any] = None

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
        on_auto_save:     Optional[Callable[[AutoSaveEvent], None]] = None,
        embedding_model:  Optional[Any] = None,
        # Backwards compat: accept on_suggestion as alias for on_auto_save
        on_suggestion:    Optional[Callable[[AutoSaveEvent], None]] = None,
        **kwargs,
    ):
        cfg   = config or RagWikiRetrieverConfig()
        store = state_store or MemoryStateStore()

        # on_suggestion is a deprecated alias for on_auto_save
        effective_callback = on_auto_save or on_suggestion

        super().__init__(
            user_id          = user_id,
            global_retriever = global_retriever,
            state_store      = store,
            config           = cfg,
            on_auto_save     = effective_callback,
            **kwargs,
        )

        sm = StateMachine()
        chunk_store = ChunkStore(wiki_save_dir=cfg.wiki_save_dir)
        resolved_model = _resolve_embedding_model(embedding_model, global_retriever)
        if resolved_model is None:
            logger.warning("Using keyword fallback (no embedding model available)")

        object.__setattr__(self, "_sm", sm)
        object.__setattr__(self, "_embedding_model", resolved_model)
        object.__setattr__(self, "_chunk_store", chunk_store)
        object.__setattr__(self, "_counter", FetchCounter(
            store               = store,
            state_machine       = sm,
            fetch_threshold     = cfg.fetch_threshold,
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
        object.__setattr__(self, "_record_cache", {
            "pinned":      None,
            "claimed":     None,
            "surfaced":    None,
            "cached_at":   None,
            "ttl_seconds": cfg.record_cache_ttl_seconds,
            "dirty":       False,
        })
        object.__setattr__(self, "_matrix_cache", {})

    @property
    def embedding_model_resolved(self) -> bool:
        """True if an embedding model was found for semantic search."""
        return object.__getattribute__(self, "_embedding_model") is not None

    # ─── LangChain interface ───────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        retrieved_meta: list[dict]               = []
        user_records:   dict[str, UserDocRecord] = {}
        auto_save:      Optional[AutoSaveEvent]  = None
        docs:           list[Document]           = []

        # ── Embed query once — reused for all cache lookups ───────────────────
        query_vec   = self._embed_query(query)
        chunk_store = object.__getattribute__(self, "_chunk_store")

        # ── Step 1+2: Unified semantic search over CLAIMED + PINNED cache ─────
        pinned, claimed, surfaced = self._get_cached_records()
        cached_records = pinned + claimed
        served_doc_ids = set()

        for record in cached_records:
            chunks, matrix = self._get_cached_matrix(record.doc_id)
            now    = datetime.now(timezone.utc)

            if query_vec is not None and chunks:
                # ── Semantic path ──────────────────────────────────────────────
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
                            self._invalidate_record_cache()

                        # Stamp last_fetched_at so decay recency signal stays alive
                        record.last_fetched_at = now
                        self.state_store.upsert(record)
                        self._invalidate_record_cache()

                        # Resolve page content — full doc or matched chunks
                        if record.always_full_doc:
                            full_text = self._resolve_full_content(record)
                            if full_text:
                                page_content      = full_text
                                full_doc_injected = True
                            else:
                                page_content      = "\n\n".join(chunks[i]["text"] for i in hit_indices)
                                full_doc_injected = False
                                logger.warning(
                                    "always_full_doc=True but content unavailable for "
                                    "doc_id=%r, falling back to chunks", record.doc_id
                                )
                        else:
                            page_content      = "\n\n".join(chunks[i]["text"] for i in hit_indices)
                            full_doc_injected = False

                        logger.debug("Semantic path: doc_id=%r matched %d chunks", record.doc_id, len(hit_indices))

                        docs.append(Document(
                            page_content = page_content,
                            metadata     = {
                                "doc_id":            record.doc_id,
                                "doc_title":         record.doc_title,
                                "doc_path":          record.doc_path,
                                "from_cache":        True,
                                "user_state":        record.user_state.value,
                                "chunks_used":       hit_indices,
                                "scores":            [float(scores[i]) for i in hit_indices],
                                "full_doc_injected": full_doc_injected,
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
                            self._invalidate_matrix_cache(record.doc_id)

                        self.state_store.upsert(record)
                        self._invalidate_record_cache()

            else:
                # ── Keyword fallback (no embedding model or no chunks yet) ─────
                if record.full_content and self._is_relevant(query, record.full_content):
                    logger.debug("Keyword path: doc_id=%r matched full content", record.doc_id)
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

        # Collect docs for batch embedding — done once after the loop
        chunks_to_embed: list[tuple[str, Document]] = []

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

            # Queue for batch embedding (one embed_documents call after loop)
            chunks_to_embed.append((doc_id, gdoc))

            docs.append(gdoc)

            # Cache content so auto-save can use it without re-fetching
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
                self._invalidate_record_cache()

                # ── Step 5: Handle auto-save if threshold crossed ──────────────
                if event is not None:
                    self._perform_auto_save(event, gdoc.page_content)
                    self._invalidate_record_cache()
                    if self.on_auto_save:
                        self.on_auto_save(event)
                    if auto_save is None:
                        auto_save = event

            # Refresh record for provenance
            record = self.state_store.get(self.user_id, doc_id)
            if record:
                user_records[doc_id] = record

        # ── Step 3b: Batch-embed all new global RAG chunks in one API call ────
        # embed_documents() is called once per query for all new chunks combined.
        self._accumulate_chunks_batch(chunks_to_embed)

        # ── Step 6: Lazy miss tracking — only upsert when streak crosses reset_threshold
        retrieved_ids = {
            m["doc_id"] for m in retrieved_meta if not m.get("from_cache")
        }
        tracked_docs = surfaced
        for record in tracked_docs:
            if record.doc_id not in retrieved_ids:
                record.queries_missed += 1
                # Only write to storage when the streak actually matters
                if record.queries_missed >= self.config.reset_threshold:
                    # Sync DB to the value just before reset so FetchCounter triggers
                    record.queries_missed -= 1
                    self.state_store.upsert(record)
                    
                    self._counter.record_miss(self.user_id, record.doc_id)
                    record.queries_missed = 0  # reset in-memory
                    self._invalidate_record_cache()
                else:
                    # Do NOT upsert on every miss — only when threshold crossed
                    pass
            else:
                if record.queries_missed > 0:
                    record.queries_missed = 0
                    self.state_store.upsert(record)  # only write if value changed
                    self._invalidate_record_cache()

        # ── Step 7: Build provenance ──────────────────────────────────────────
        provenance = self._builder.build(
            retrieved_docs = retrieved_meta,
            user_records   = user_records,
            auto_save      = auto_save,
            embedding_model_resolved = self.embedding_model_resolved,
        )
        object.__setattr__(self, "last_provenance", provenance)

        return docs

    # ─── User action API ──────────────────────────────────────────────────────

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

    def set_always_full_doc(self, doc_id: str, enabled: bool = True) -> UserDocRecord:
        """
        Mark a CLAIMED or PINNED document for full-document injection.

        When enabled=True, any cache hit injects the entire document text
        instead of only the matching chunks. The similarity check still runs
        to confirm relevance before injection.

        Only valid for CLAIMED or PINNED documents.
        Raises ValueError if the document is not in the personal cache.
        """
        record = self.state_store.get(self.user_id, doc_id)
        if record is None:
            raise ValueError(
                f"No cached record for user={self.user_id!r}, doc={doc_id!r}. "
                "Document must be CLAIMED or PINNED before setting always_full_doc."
            )
        if record.user_state not in (DocumentState.CLAIMED, DocumentState.PINNED):
            raise ValueError(
                f"always_full_doc can only be set on CLAIMED or PINNED documents. "
                f"Current state: {record.user_state.value}"
            )
        record.always_full_doc = enabled
        self.state_store.upsert(record)
        return record

    # ─── Private ──────────────────────────────────────────────────────────────

    def _perform_auto_save(self, event: AutoSaveEvent, fallback_content: str) -> None:
        """
        Automatically save a document after it crosses the fetch threshold.
        Resolves content, writes a wiki copy, and chunks+embeds the doc.
        """
        content = None

        # Try to read the full file from disk using the stored doc_path
        record = self.state_store.get(self.user_id, event.doc_id)
        doc_path = record.doc_path if record else ""
        if doc_path and os.path.isfile(doc_path):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError as exc:
                logger.warning("Could not read doc_path=%r: %s", doc_path, exc)

        # Fall back to cached chunk content
        if not content:
            content_cache = object.__getattribute__(self, "_content_cache")
            content = content_cache.get(event.doc_id, fallback_content)

        # Store full content on the record
        if record and content:
            record.full_content = content
            self.state_store.upsert(record)

        # Optionally save a copy to wiki_save_dir
        if content and self.config.wiki_save_dir:
            try:
                os.makedirs(self.config.wiki_save_dir, exist_ok=True)
                doc_title = event.doc_title
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                safe_title = "".join(
                    c if c.isalnum() or c in " ._-" else "_" for c in doc_title
                )
                wiki_filename = f"{timestamp}_{safe_title}"
                wiki_path = os.path.join(self.config.wiki_save_dir, wiki_filename)
                with open(wiki_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info("Auto-saved doc copy to %s", wiki_path)
            except OSError as exc:
                logger.warning("Could not save doc copy to wiki_save_dir: %s", exc)

        # Embed and store all chunks
        if content:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                doc_to_split = Document(page_content=content, metadata={"doc_id": event.doc_id})
                split_docs = splitter.split_documents([doc_to_split])

                for i, doc in enumerate(split_docs):
                    doc.metadata["chunk_index"] = i

                self._accumulate_chunks_batch([(event.doc_id, doc) for doc in split_docs])
                self._invalidate_matrix_cache(event.doc_id)
                logger.info(
                    "Auto-save: chunked, embedded and saved %d chunks for doc_id=%r",
                    len(split_docs), event.doc_id,
                )
            except Exception as exc:
                logger.warning("Failed to chunk and embed doc_id=%r: %s", event.doc_id, exc)

    def _resolve_full_content(self, record: UserDocRecord) -> Optional[str]:
        """
        Resolve full document text for always_full_doc injection.

        Priority:
          1. record.full_content (already in memory/DB)
          2. Read from record.doc_path if it is a readable file
          3. Return None (caller handles fallback)
        """
        if record.full_content:
            return record.full_content
        if record.doc_path and os.path.isfile(record.doc_path):
            try:
                with open(record.doc_path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError as exc:
                logger.warning(
                    "Could not read doc_path=%r for always_full_doc: %s",
                    record.doc_path, exc,
                )
        return None

    def _get_cached_records(self) -> tuple[list, list, list]:
        """Return (pinned, claimed, surfaced) from cache or re-fetch if stale."""
        rc  = object.__getattribute__(self, "_record_cache")
        now = datetime.now(timezone.utc)

        if (
            rc["pinned"] is not None
            and rc["claimed"] is not None
            and rc["surfaced"] is not None
            and not rc["dirty"]
            and rc["cached_at"] is not None
            and (now - rc["cached_at"]).total_seconds() < rc["ttl_seconds"]
        ):
            return rc["pinned"], rc["claimed"], rc["surfaced"]

        pinned   = self.state_store.list_pinned(self.user_id)
        claimed  = self.state_store.list_claimed(self.user_id)
        surfaced = self.state_store.list_surfaced(self.user_id)
        rc.update({
            "pinned": pinned, "claimed": claimed, "surfaced": surfaced,
            "cached_at": now, "dirty": False,
        })
        return pinned, claimed, surfaced

    def _invalidate_record_cache(self) -> None:
        """Mark record cache as stale so next query re-fetches."""
        rc = object.__getattribute__(self, "_record_cache")
        rc["dirty"] = True

    def _get_cached_matrix(
        self, doc_id: str
    ) -> tuple[list[dict], Optional[np.ndarray]]:
        """Return (chunks, matrix) from cache or re-load from ChunkStore."""
        mc  = object.__getattribute__(self, "_matrix_cache")
        key = (self.user_id, doc_id)
        if key in mc and not mc[key]["dirty"]:
            return mc[key]["chunks"], mc[key]["matrix"]

        chunk_store = object.__getattribute__(self, "_chunk_store")
        chunks = chunk_store.load_chunks(self.user_id, doc_id)
        matrix = ChunkStore.build_matrix(chunks) if chunks else None
        mc[key] = {"chunks": chunks, "matrix": matrix, "dirty": False}
        return chunks, matrix

    def _invalidate_matrix_cache(self, doc_id: str) -> None:
        """Mark matrix cache entry as stale for next query."""
        mc  = object.__getattribute__(self, "_matrix_cache")
        key = (self.user_id, doc_id)
        if key in mc:
            mc[key]["dirty"] = True

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
            self._invalidate_matrix_cache(doc_id)



    def _is_relevant(self, query: str, content: str) -> bool:
        """
        Lightweight keyword overlap check for cache hits when no embedding
        model is available.
        """
        query_terms   = set(query.lower().split())
        content_lower = content.lower()
        matches       = sum(1 for term in query_terms if term in content_lower)
        return matches >= max(1, len(query_terms) // 3)
