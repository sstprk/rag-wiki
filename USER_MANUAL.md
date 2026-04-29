# Rag-Wiki User Manual

Welcome to `rag-wiki`! This package provides a LangChain-compatible retriever that seamlessly blends your existing "global" knowledge base (like a Vector DB) with personalized, user-specific cached documents.

This manual explains how to integrate, configure, and operate the system in a production environment.

---

## Overview of the Architecture

`rag-wiki` is designed as a LangChain `BaseRetriever`. You drop it in exactly where you would normally use a standard vector store retriever.

When a user submits a query:
1. **Pinned Docs** — any document explicitly pinned by the user is automatically injected into the context.
2. **Cache Hit** — if the query matches a document the user has previously claimed (saved), it is retrieved directly from local fast storage, skipping the vector search entirely.
3. **Global Fallback** — if the cache misses, the query is passed to your underlying global retriever (e.g., a Chroma or LlamaIndex vector store).

Behind the scenes, `rag-wiki` tracks which documents are being fetched, handles the lifecycle of suggesting documents to users, and decays their relevance over time.

---

## 1. Setting Up Storage

You need a `StateStore` to track document states. The package provides three options.

### `MemoryStateStore` (Development / Testing)
Zero dependencies, keeps data in memory. Perfect for prototyping. Thread-safe. Data is lost when the process exits.
```python
from rag_wiki import MemoryStateStore
store = MemoryStateStore()
```

### `SQLiteStateStore` (Single Node Production)
Persists data to an SQLite file. Great for single-server deployments. The recommended path is inside the `wiki/` folder so all persistent state lives in one place.
```python
from rag_wiki.storage.sqlite import SQLiteStateStore

store = SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db")
```

### `RedisStateStore` (Distributed Production)
Uses Redis to store data. Essential if you are running multiple API workers or load-balancing across servers.
```python
import redis
from rag_wiki.storage.redis_store import RedisStateStore

client = redis.Redis(host="localhost", port=6379, db=0)
store = RedisStateStore(client)
```

---

## 2. Integrating the Retriever

Wrap your existing retriever with `RagWikiRetriever`. Specify the `user_id` for each operation.

```python
import os
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig
from rag_wiki.storage.sqlite import SQLiteStateStore

os.makedirs("wiki/documents", exist_ok=True)

global_retriever = my_vector_db.as_retriever(search_kwargs={"k": 5})

retriever = RagWikiRetriever(
    user_id="user-123",
    global_retriever=global_retriever,
    state_store=SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db"),
    config=RagWikiRetrieverConfig(
        fetch_threshold=3,   # queries before first suggestion
        reset_threshold=5,   # missed queries before fetch count resets
    ),
)

docs = retriever.invoke("How do I configure Kubernetes pods?")
```

### Transparency & Provenance
After calling `.invoke()`, you can inspect where each document came from:
```python
print(retriever.last_provenance.render())
# ────────────────────────────────────────────────────────────
# 📄 Sources used in this response
#   • Kubernetes Guide [from your KB]
#     Full document  |  Saved to your KB
#   • Pod Networking
#     Full document  |  SURFACED (fetched 2×)
# ────────────────────────────────────────────────────────────
```

---

## 3. The `wiki/` Folder

All persistent data is stored under `wiki/` in your project root:

```
wiki/
├── documents/          ← full copies of saved documents (timestamped)
└── rag_wiki_state.db   ← SQLite state store
```

When a user accepts a suggestion, the full source file is read from disk and:
- stored in the SQLite DB as `full_content` for fast cache retrieval
- saved as a timestamped copy in `wiki/documents/` (e.g. `20260429_131526_MyDoc.txt`)

Make sure to create the folder before starting:
```python
import os
os.makedirs("wiki/documents", exist_ok=True)
```

---

## 4. The UI Workflow: Suggestions

The core value of `rag-wiki` is suggesting documents to users so they can save them for fast cached access.

### How Suggestions Are Triggered

When a document is retrieved from the global RAG enough times to cross `fetch_threshold`, a `SuggestionEvent` is fired via the `on_suggestion` callback.

### Escalating Re-suggestion After Decline

If the user declines, the system doesn't give up — it schedules the next suggestion further out, doubling the gap each time:

| Decline # | Suggestion fires at fetch count |
|-----------|--------------------------------|
| First suggestion | `fetch_threshold` (e.g. 2) |
| After 1st decline | `fetch_count + threshold × 2` (e.g. 6) |
| After 2nd decline | `fetch_count + prev_gap × 2` (e.g. 14) |
| After 3rd decline | (e.g. 30) |

### Automatic Threshold Reset

If a document stops appearing in queries for `reset_threshold` consecutive queries, its fetch count and suggestion target are reset to zero. This prevents stale counts from triggering suggestions for documents the user no longer encounters.

```python
RagWikiRetrieverConfig(
    fetch_threshold=2,   # suggest after 2 fetches
    reset_threshold=3,   # reset if not seen for 3 queries in a row
)
```

### Handling Suggestions in Code

```python
pending = []

def on_suggestion(event):
    pending.append(event)

retriever = RagWikiRetriever(
    user_id="user-123",
    global_retriever=global_retriever,
    state_store=store,
    on_suggestion=on_suggestion,
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

> **Note:** Multiple suggestions can fire in a single query (one per document that crosses the threshold). The loop above handles all of them.

### Accepting a Suggestion

```python
retriever.accept_suggestion(doc_id="doc-789")
```

`accept_suggestion` automatically:
1. Reads the full file content from the original `doc_path` on disk.
2. Stores the full content in the state DB for future cache hits.
3. Saves a timestamped copy to `wiki/documents/`.
4. Falls back to the cached chunk if the original file is not readable.

You can also pass content explicitly:
```python
retriever.accept_suggestion(doc_id="doc-789", full_content="The full text...")
```

### Declining a Suggestion

```python
retriever.decline_suggestion(doc_id="doc-789")
# Next suggestion will be scheduled further out (escalating gap).
```

---

## 5. User Interactions & Explicit Signals

Users can explicitly manage their saved documents at any time.

**Thumbs up / down** (affects decay score):
```python
retriever.thumbs_up(doc_id="doc-789")
retriever.thumbs_down(doc_id="doc-789")
```

**Force actions**:
```python
retriever.force_pin(doc_id="doc-789")    # always inject this document into every query
retriever.force_remove(doc_id="doc-789") # remove from personal KB entirely
```

---

## 6. Background Decay Jobs

To keep the personal KB clean, documents decay over time if they aren't used. Run the `DecayScheduler` in the background.

For FastAPI/Flask in a single process, the threaded scheduler is easiest:

```python
from rag_wiki.lifecycle.scheduler import DecayScheduler

scheduler = DecayScheduler(store, backend="simple")
scheduler.start()
# Call scheduler.stop() on shutdown
```

For `APScheduler` or more complex environments:
```python
scheduler = DecayScheduler(store, backend="apscheduler")
scheduler.start()
```

> In a distributed environment with Redis, ensure only **one** instance of the scheduler runs across your cluster, or use a distributed lock.

---

## 7. Configuration Reference

```python
RagWikiRetrieverConfig(
    fetch_threshold=3,       # fetches before first suggestion fires
    reset_threshold=3,       # consecutive missed queries before fetch count resets to 0
    no_resiluggest_days=30,  # legacy field, kept for API compatibility
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fetch_threshold` | `3` | Number of times a doc must be retrieved before the first suggestion |
| `reset_threshold` | `3` | Consecutive queries without a hit before fetch count resets |
| `no_resiluggest_days` | `30` | Kept for API compatibility (escalating gap logic replaces this) |
