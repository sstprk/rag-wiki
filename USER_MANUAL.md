# Rag-Wiki User Manual

Welcome to `rag-wiki`! This package provides a LangChain-compatible retriever that seamlessly blends your existing "global" knowledge base (like a Vector DB) with personalized, user-specific cached documents. 

This manual explains how to integrate, configure, and operate the system in a production environment.

## Overview of the Architecture

`rag-wiki` is designed as a LangChain `BaseRetriever`. You drop it in exactly where you would normally use a standard vector store retriever.

When a user submits a query:
1. **Pinned Docs**: Any document explicitly pinned by the user is automatically injected into the context.
2. **Cache Hit**: If the query matches a document the user has previously claimed (saved), it is retrieved directly from local fast storage.
3. **Global Fallback**: If the cache misses, the query is passed to your underlying global retriever (e.g., LlamaIndex or LangChain VectorStore).

Behind the scenes, `rag-wiki` tracks exactly which documents are being fetched and handles the lifecycle of suggesting documents to users and decaying their relevance over time.

---

## 1. Setting Up Storage

You need a `StateStore` to track document states. The package provides three options:

### `MemoryStateStore` (Development / Testing)
Zero dependencies, keeps data in memory. Perfect for prototyping. Thread-safe.
```python
from rag_wiki import MemoryStateStore
store = MemoryStateStore()
```

### `SQLiteStateStore` (Single Node Production)
Persists data to an SQLite file. Great for single-server deployments like a simple FastAPI app.
```python
from rag_wiki.storage.sqlite import SQLiteStateStore
store = SQLiteStateStore("sqlite:///rag_wiki.db")
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

Wrap your existing retriever with `RagWikiRetriever`. You must create a new instance for **every user request** because LangChain retrievers are generally stateless, and you need to specify the `user_id` for each operation.

```python
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig

# 1. Have your global retriever ready
global_retriever = my_vector_db.as_retriever(search_kwargs={"k": 5})

# 2. Instantiate the hybrid retriever for the current user
retriever = RagWikiRetriever(
    user_id="user-123",
    global_retriever=global_retriever,
    state_store=store, # From step 1
    config=RagWikiRetrieverConfig(fetch_threshold=3),
)

# 3. Use it exactly like a normal LangChain retriever!
docs = retriever.invoke("How do I configure Kubernetes pods?")
```

### Transparency & Provenance
After calling `.invoke()`, you can check *where* the documents came from:
```python
print(retriever.last_provenance.render())
# Output:
# Sources used:
#  - Kubernetes Guide [Global RAG]
#  - Pod Networking [Cache Hit]
```

---

## 3. The UI Workflow: Suggestions

The core value of `rag-wiki` is suggesting documents to users so they can "save" them for fast offline/cached access.

When a user repeatedly fetches the same document from the global RAG (exceeding `fetch_threshold`), a suggestion is triggered.

### Handling Suggestions
You can listen for suggestions by providing a callback:

```python
def handle_suggestion(event):
    # Send a WebSocket message, an email, or return an API flag to the frontend
    print(f"Hey User! Want to save '{event.doc_title}' to your personal KB?")

retriever = RagWikiRetriever(
    user_id="user-123",
    global_retriever=global_retriever,
    state_store=store,
    on_suggestion=handle_suggestion
)
```

### Accepting or Declining
When your frontend asks the user and they respond, you use the retriever to apply their decision:

**If they accept (Save):**
```python
# Provide the full content so it can be cached locally!
retriever.accept_suggestion(doc_id="doc-789", full_content="The full markdown text...")
```

**If they decline (Not right now):**
```python
retriever.decline_suggestion(doc_id="doc-789")
# The system won't suggest this document again for 30 days.
```

---

## 4. User Interactions & Explicit Signals

Users might want to explicitly manage their saved documents.

**Thumb Up/Down** (Affects decay score):
```python
retriever.thumbs_up(doc_id="doc-789")
retriever.thumbs_down(doc_id="doc-789")
```

**Force Actions**:
```python
retriever.force_pin(doc_id="doc-789")    # Always inject this document
retriever.force_remove(doc_id="doc-789") # Remove from personal KB
```

---

## 5. Background Decay Jobs

To keep the personal KB clean, documents "decay" over time if they aren't used. You must run the `DecayScheduler` in the background.

If you are using FastAPI/Flask in a single process, the `simple` threaded scheduler is easiest:

```python
from rag_wiki.lifecycle.scheduler import DecayScheduler

# Start this when your app starts
scheduler = DecayScheduler(store, backend="simple")
scheduler.start()

# Call scheduler.stop() when your app shuts down
```

If you prefer `APScheduler` or are running in a more complex environment:
```python
scheduler = DecayScheduler(store, backend="apscheduler")
scheduler.start()
```

*(Note: In a distributed environment with Redis, ensure you only run ONE instance of the scheduler across your cluster, or use a distributed lock).*
