# Rag-Wiki User Manual

Welcome to `rag-wiki`! This package provides a LangChain-compatible retriever that blends your existing global knowledge base with a personal, user-curated cache that gets smarter with every query.

---

## Overview of the Architecture

`rag-wiki` is designed as a LangChain `BaseRetriever`. Drop it in wherever you would use a standard vector store retriever.

When a user submits a query:

1. **Semantic cache search** — PINNED and CLAIMED documents are searched using cosine similarity against their stored chunk vectors. Only chunks that score above `similarity_threshold` are injected, ranked by score, capped at `local_top_k`. If no embedding model is available, falls back to keyword matching.
2. **Global RAG fallback** — if the cache misses (or the doc isn't claimed yet), the query goes to your underlying vector retriever.

When a user accepts a suggestion, the full document is locally chunked and embedded in the background, creating the chunk index for subsequent fast semantic cache searches.

---

## 1. Setting Up Storage

You need a `StateStore` to track document states. The package provides three options.

### `MemoryStateStore` (Development / Testing)
Zero dependencies, keeps data in memory. Thread-safe. Data is lost when the process exits.
```python
from rag_wiki import MemoryStateStore
store = MemoryStateStore()
```

### `SQLiteStateStore` (Single Node Production)
Persists to an SQLite file. The recommended path is inside the `wiki/` folder so all persistent state lives in one place.
```python
from rag_wiki.storage.sqlite import SQLiteStateStore

store = SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db")
```

### `RedisStateStore` (Distributed Production)
Required when running multiple API workers or load-balancing across servers.
```python
import redis
from rag_wiki.storage.redis_store import RedisStateStore

client = redis.Redis(host="localhost", port=6379, db=0)
store = RedisStateStore(client)
```

---

## 2. Integrating the Retriever

```python
import os
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig
from rag_wiki.storage.sqlite import SQLiteStateStore

os.makedirs("wiki/documents", exist_ok=True)

retriever = RagWikiRetriever(
    user_id          = "user-123",
    global_retriever = my_vector_db.as_retriever(search_kwargs={"k": 5}),
    state_store      = SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db"),
    config           = RagWikiRetrieverConfig(
        fetch_threshold      = 3,      # queries before first suggestion
        reset_threshold      = 5,      # missed queries before fetch count resets
        similarity_threshold = 0.75,   # cosine similarity cutoff for cache hits
        local_top_k          = 3,      # max chunks injected per cached doc
        wiki_save_dir        = "wiki/documents",
    ),
)

docs = retriever.invoke("How do I configure Kubernetes pods?")
print(retriever.last_provenance.render())
```

### Embedding Model

The embedding model used for semantic cache search is resolved automatically in this order:

1. `embedding_model` argument if passed explicitly
2. Auto-extracted from `global_retriever` via common attribute names (`embeddings`, `embedding_function`, etc.)
3. `None` — keyword fallback is used silently, no error

For most LangChain vector stores (Chroma, FAISS, etc.) the model is auto-resolved from the retriever. You only need to pass it explicitly if your retriever doesn't expose it:

```python
from langchain_openai import OpenAIEmbeddings

retriever = RagWikiRetriever(
    user_id          = "user-123",
    global_retriever = your_retriever,
    embedding_model  = OpenAIEmbeddings(),  # explicit override
)
```

### Transparency & Provenance

After every `.invoke()`, inspect where each document came from:

```python
print(retriever.last_provenance.render())
# ────────────────────────────────────────────────────────────
# 📄 Sources used in this response
#   • Kubernetes Guide [from your KB]
#     Chunks 0, 2  |  Saved to your KB
#   • Pod Networking
#     Full document  |  SURFACED (fetched 2×)
# ────────────────────────────────────────────────────────────
```

Also available as a dict for API consumers:
```python
data = retriever.last_provenance.to_dict()
```

---

## 3. The `wiki/` Folder

All persistent data lives under `wiki/` in your project root:

```
wiki/
├── documents/              ← timestamped copies of accepted docs
│   └── <user_id>/
│       └── <doc_id>/
│           └── chunks.jsonl   ← chunk vectors for semantic search
└── rag_wiki_state.db       ← SQLite state store
```

The `chunks.jsonl` file is the chunk-level vector index. Each line is one chunk:

```json
{"chunk_index": 0, "text": "...", "vector": [0.02, ...], "section": "", "hit_count": 3, "last_hit_at": "2026-04-30T..."}
```

Chunks are generated automatically when a document is claimed. Upon acceptance, the system locally chunks the full document and embeds it via a single batch call (`embed_documents`), so chunking an entire document costs just one API call. You don't need to manage this file directly.

---

## 4. The UI Workflow: Suggestions

### How Suggestions Are Triggered

When a document is retrieved from global RAG `fetch_threshold` times, a `SuggestionEvent` fires via the `on_suggestion` callback.

### Escalating Re-suggestion After Decline

If the user declines, the next suggestion is scheduled further out, doubling the gap each time:

| Decline # | Suggestion fires at fetch count |
|-----------|--------------------------------|
| First suggestion | `fetch_threshold` (e.g. 3) |
| After 1st decline | `fetch_count + threshold × 2` (e.g. 9) |
| After 2nd decline | `fetch_count + prev_gap × 2` (e.g. 21) |
| After 3rd decline | (e.g. 45) |

### Automatic Threshold Reset

If a document stops appearing in queries for `reset_threshold` consecutive queries, its fetch count resets to 0. This prevents stale counts from triggering suggestions for documents the user no longer encounters.

### Handling Suggestions in Code

```python
pending = []

def on_suggestion(event):
    pending.append(event)

retriever = RagWikiRetriever(
    user_id       = "user-123",
    global_retriever = global_retriever,
    state_store   = store,
    on_suggestion = on_suggestion,
)

docs = retriever.invoke("your query")

for event in pending:
    print(f"💡 '{event.doc_title}' has come up {event.fetch_count}× — save to library? [y/n]")
    if input().strip().lower() == "y":
        retriever.accept_suggestion(event.doc_id)
    else:
        retriever.decline_suggestion(event.doc_id)

pending.clear()
```

Multiple suggestions can fire in a single query — one per document that crosses the threshold. The loop handles all of them.

### Accepting a Suggestion

```python
retriever.accept_suggestion(doc_id="doc-789")
```

Content resolution order:
1. `full_content` argument if provided
2. Full file read from `doc_path` on disk
3. Cached chunk text from last retrieval (fallback)

The chunk vector index (`chunks.jsonl`) is automatically built at this step — the document is chunked and embedded in the background.

If `wiki_save_dir` is configured, a timestamped copy is written there automatically.

### Declining a Suggestion

```python
retriever.decline_suggestion(doc_id="doc-789")
# Next suggestion scheduled at escalating interval.
```

---

## 5. Semantic Cache Behaviour

### How It Works

When a CLAIMED or PINNED document is in the cache, the retriever:

1. Loads the document's `chunks.jsonl`
2. Builds a matrix of chunk vectors
3. Computes cosine similarity between the query vector and every chunk
4. Injects the top `local_top_k` chunks that score ≥ `similarity_threshold`

Cosine similarity is always computed on normalised vectors, so scores are in `[-1, 1]` regardless of the embedding model's output magnitude.

### Cache Miss Streak and Auto-Demotion

Every time a cached document's chunks all score below `similarity_threshold`, its `cache_miss_streak` counter increments. When it reaches `max_cache_miss_streak` (default: 10), the document is demoted immediately:

- State transitions to `DEMOTED`
- `chunks.jsonl` is deleted
- Document returns to the global vector search pool

This prevents token bloat from cached documents that are no longer relevant to the user's current queries.

### Keyword Fallback

If no embedding model is available (neither auto-resolved nor explicitly passed), the cache falls back to keyword overlap matching. This is less precise but ensures the system always works regardless of infrastructure.

---

## 6. User Interactions & Explicit Signals

```python
# Boost decay score — user found this useful
retriever.thumbs_up(doc_id="doc-789")

# Reduce decay score — user found this not useful
retriever.thumbs_down(doc_id="doc-789")

# Always inject this document into every query context
retriever.force_pin(doc_id="doc-789")

# Remove from personal KB entirely (also deletes chunk index)
retriever.force_remove(doc_id="doc-789")
```

---

## 7. Background Decay Jobs

Documents decay over time if they aren't used. Run the `DecayScheduler` in the background to keep states current.

```python
from rag_wiki import DecayEngine, DecayConfig
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.scheduler import DecayScheduler
from rag_wiki.storage.sqlite import SQLiteStateStore
from rag_wiki.storage.chunk_store import ChunkStore

store       = SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db")
chunk_store = ChunkStore(wiki_save_dir="wiki")
engine      = DecayEngine(
    store, StateMachine(),
    config=DecayConfig(),
    chunk_store=chunk_store,   # enables chunk hit rate in decay score
)
scheduler = DecayScheduler(engine, store, backend="simple", interval_hours=24)
scheduler.start()
# Call scheduler.stop() on shutdown
```

Use `backend="apscheduler"` for production (requires `pip install 'langchain-rag-wiki[scheduler]'`).

The decay score formula:

```
decay_score = weighted_avg(
    recency_factor   = exp(-λ × days_since_last_fetch),   weight: 0.40
    frequency_factor = min(fetch_count / freq_cap, 1.0),  weight: 0.30
    explicit_signal  = thumbs_up / thumbs_down value,     weight: 0.20
    chunk_hit_rate   = fraction of chunks ever matched,   weight: 0.15
)
```

Documents above `pin_threshold` (0.85) are auto-pinned after `pin_hold_days`. Documents below `demotion_threshold` (0.15) are evicted after `demotion_hold_days`. When a demotion fires, the chunk index is deleted automatically.

In a distributed environment with Redis, ensure only **one** scheduler instance runs across your cluster, or use a distributed lock.

---

## 8. Configuration Reference

### `RagWikiRetrieverConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fetch_threshold` | `3` | Fetches before first suggestion fires |
| `reset_threshold` | `3` | Consecutive missed queries before fetch count resets |
| `similarity_threshold` | `0.75` | Cosine similarity cutoff for cache chunk hits |
| `local_top_k` | `3` | Max chunks injected per cached doc per query |
| `wiki_save_dir` | `None` | Directory to save accepted doc copies; `None` disables |
| `no_resiluggest_days` | `30` | Deprecated — kept for API compatibility |
| `decay` | `DecayConfig()` | Decay engine settings (see below) |

### `DecayConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_recency` | `0.40` | Weight for recency factor |
| `w_frequency` | `0.30` | Weight for frequency factor |
| `w_explicit` | `0.20` | Weight for explicit user signals |
| `w_chunk_hit` | `0.15` | Weight for chunk hit rate |
| `max_cache_miss_streak` | `10` | Consecutive cache misses before immediate demotion |
| `decay_lambda` | `0.05` | Decay steepness λ (half-life ≈ 14 days) |
| `freq_cap` | `20` | Max fetch count for frequency normalisation |
| `pin_threshold` | `0.85` | Score above which doc is auto-pinned |
| `demotion_threshold` | `0.15` | Score below which doc is auto-demoted |
| `pin_hold_days` | `7` | Days score must hold above threshold before pin fires |
| `demotion_hold_days` | `3` | Days score must hold below threshold before demotion fires |
