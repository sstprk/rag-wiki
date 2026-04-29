<div align="center">

# langchain-rag-wiki

**RAG meets wiki-style provenance.**  
Surfaces documents, learns from usage, and lets users build personal knowledge bases on top of shared vector stores.

[![PyPI version](https://img.shields.io/pypi/v/langchain-rag-wiki.svg)](https://pypi.org/project/langchain-rag-wiki/) 
[![Python](https://img.shields.io/pypi/pyversions/langchain-rag-wiki.svg)](https://pypi.org/project/langchain-rag-wiki/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTE.md)

[Installation](#installation) · [Quickstart](#quickstart) · [How It Works](#how-it-works) · [Real Vector DB Setup](#connecting-to-a-real-vector-db) · [Configuration](#configuration) · [Contributing](CONTRIBUTING.md)

</div>

---

## What Is This?

Standard RAG is stateless — every query hits the vector database fresh, returns probabilistic results, and forgets what was useful last time.

`langchain-rag-wiki` wraps any LangChain retriever and adds a personal knowledge layer on top:

- Documents retrieved repeatedly get **suggested for saving** after a configurable threshold
- Once saved, they're served from a **local cache** — no vector search, deterministic results
- Every response shows a **provenance block** so you always know which document was used and from where
- A **decay model** keeps the cache honest — stale documents fade out automatically

It's a drop-in `BaseRetriever`. Any LangChain chain, agent, or RAG pipeline that accepts a retriever works with it immediately.

---

## Installation

```bash
pip install langchain-rag-wiki
```

With optional backends:

```bash
pip install 'langchain-rag-wiki[sqlite]'     # SQLite persistent store
pip install 'langchain-rag-wiki[redis]'       # Redis distributed store
pip install 'langchain-rag-wiki[scheduler]'   # APScheduler decay jobs
pip install 'langchain-rag-wiki[llama]'       # LlamaIndex adapter
pip install 'langchain-rag-wiki[dev]'         # all of the above + pytest + fakeredis
```

---

## Quickstart

Zero infrastructure — uses in-memory state. No API keys needed.

```python
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig, MemoryStateStore

retriever = RagWikiRetriever(
    user_id          = "user-123",
    global_retriever = your_existing_retriever,  # any LangChain BaseRetriever
    state_store      = MemoryStateStore(),
    config           = RagWikiRetrieverConfig(fetch_threshold=3),
)

docs = retriever.invoke("quarterly earnings report")
print(retriever.last_provenance.render())  # see what was retrieved and from where
```

Run the bundled demo (no API keys, no services required):

```bash
python example.py
```

---

## How It Works

Every query goes through three retrieval tiers in order:

```
1. PINNED docs    → always injected into context (user explicitly pinned these)
2. CLAIMED cache  → direct local lookup, skips vector search entirely
3. Global RAG     → fallback to your existing vector retriever
```

### Document Lifecycle

Documents move through six states based on usage patterns:

```
GLOBAL → SURFACED → SUGGESTED → CLAIMED → PINNED
                ↘               ↙
                  DEMOTED ──────
```

| State | What it means | Retrieval path |
|-------|--------------|----------------|
| `GLOBAL` | In the shared vector DB only | Vector similarity search |
| `SURFACED` | Retrieved at least once; counter is active | Vector search (counter increments) |
| `SUGGESTED` | Fetch count ≥ threshold; user prompted once | Vector search (pending decision) |
| `CLAIMED` | User saved it; full doc in local cache | Direct local lookup |
| `PINNED` | Consistently relevant; auto-promoted | Always injected into context |
| `DEMOTED` | Usage dropped; evicted from cache | Returns to vector search |

### Threshold and Suggestion

The save suggestion fires **once** when a document has been retrieved `fetch_threshold` times (default: 3). If the user declines, it won't re-surface for `no_resiluggest_days` (default: 30 days). The `reset_threshold` setting resets the fetch count if the document hasn't appeared in that many consecutive queries — so only genuinely recurring documents get suggested.

### Decay Model

A background job recomputes each document's relevance score daily:

```
decay_score = weighted_avg(
    recency_factor   = exp(-λ × days_since_last_fetch),
    frequency_factor = min(fetch_count / freq_cap, 1.0),
    explicit_signal  = thumbs_up / thumbs_down value,
)
```

Documents above `pin_threshold` (0.85) get auto-pinned after `pin_hold_days`. Documents below `demotion_threshold` (0.15) get evicted after `demotion_hold_days`.

### Provenance Block

After every query, `retriever.last_provenance.render()` outputs:

```
────────────────────────────────────────────────────────────
📄 Sources used in this response
  • Kubernetes Pod Basics [from your KB]
    Chunks 0, 2 of 5  |  Saved to your KB
  • Docker Image Guide
    Chunks 1 of 8  |  SURFACED (fetched 2×)

💡 "Docker Image Guide" has appeared in your queries 3 times.
   Would you like to save it to your personal knowledge base?

   [ Save to my KB ]   [ Not now ]
────────────────────────────────────────────────────────────
```

The provenance block is also available as a structured dict via `retriever.last_provenance.to_dict()` for API consumers.

---

## Connecting to a Real Vector DB

### Ingesting Documents

Use the included `ingest.py` as a starting point. It loads `.txt` files from a folder, splits them into chunks, injects the required metadata, and stores them in Chroma using Ollama embeddings:

```bash
python ingest.py
```

The metadata fields `doc_id`, `doc_title`, and `doc_path` on each chunk are what connect the vector store to the lifecycle tracking system. Without them, fetch counting and cache promotion won't work. If you write your own ingestion pipeline, make sure every chunk carries these fields.

### Chroma + Ollama (fully local, no API key)

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag_wiki import RagWikiRetriever, MemoryStateStore

vectorstore = Chroma(
    collection_name    = "my_docs",
    embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory  = "./chroma_db",
)

retriever = RagWikiRetriever(
    user_id          = "user-1",
    global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
    state_store      = MemoryStateStore(),
)

docs = retriever.invoke("your query here")
print(retriever.last_provenance.render())
```

### Chroma + OpenAI

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rag_wiki import RagWikiRetriever, MemoryStateStore

vectorstore = Chroma(
    collection_name    = "my_docs",
    embedding_function = OpenAIEmbeddings(),
    persist_directory  = "./chroma_db",
)

retriever = RagWikiRetriever(
    user_id          = "user-1",
    global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
    state_store      = MemoryStateStore(),
)
```

### Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from rag_wiki import RagWikiRetriever, MemoryStateStore

vectorstore = PineconeVectorStore(
    index_name = "my-index",
    embedding  = OpenAIEmbeddings(),
)

retriever = RagWikiRetriever(
    user_id          = "user-1",
    global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
    state_store      = MemoryStateStore(),
)
```

---

## User Actions

After retrieving documents, users can provide explicit signals:

```python
# User confirmed: save this document to personal KB
retriever.accept_suggestion(doc_id="doc-1", full_content="full document text")

# User dismissed: don't ask again for 30 days
retriever.decline_suggestion(doc_id="doc-1")

# Explicit positive signal — boosts decay score
retriever.thumbs_up(doc_id="doc-1")

# Explicit negative signal — reduces decay score
retriever.thumbs_down(doc_id="doc-1")

# Always include this document in every query context
retriever.force_pin(doc_id="doc-1")

# Remove from personal KB entirely
retriever.force_remove(doc_id="doc-1")
```

---

## Configuration

### `RagWikiRetrieverConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fetch_threshold` | `int` | `3` | Fetch count before save suggestion fires |
| `reset_threshold` | `int` | `3` | Queries without a hit before fetch count resets |
| `no_resiluggest_days` | `int` | `30` | Days to block re-suggestion after a decline |
| `wiki_save_dir` | `str \| None` | `None` | Directory to save accepted docs to disk; `None` disables |
| `decay` | `DecayConfig` | *(see below)* | Decay engine settings |

### `DecayConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_recency` | `float` | `0.5` | Weight for recency in decay score |
| `w_frequency` | `float` | `0.3` | Weight for frequency |
| `w_explicit` | `float` | `0.2` | Weight for explicit user signals |
| `decay_lambda` | `float` | `0.05` | Decay steepness λ (half-life ≈ 14 days) |
| `freq_cap` | `int` | `20` | Max fetch count for frequency normalisation |
| `pin_threshold` | `float` | `0.85` | Score above which doc is auto-pinned |
| `demotion_threshold` | `float` | `0.15` | Score below which doc is auto-demoted |
| `pin_hold_days` | `int` | `7` | Days score must hold above threshold before pin fires |
| `demotion_hold_days` | `int` | `3` | Days score must hold below threshold before demotion fires |

---

## Storage Backends

### Memory (default, zero dependencies)

```python
from rag_wiki import MemoryStateStore
store = MemoryStateStore()
```

Thread-safe dict. Data lost on process restart. Best for development and testing.

### SQLite (single-node persistent)

```python
from rag_wiki.storage.sqlite import SQLiteStateStore
store = SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db")
```

Persists to a local file. Good for single-server production deployments.

### Redis (distributed)

```python
import redis
from rag_wiki.storage.redis_store import RedisStateStore

client = redis.Redis(host="localhost", port=6379, db=0)
store = RedisStateStore(client)
```

Required when running multiple API workers or load-balancing. Uses Redis hashes and sets for fast state queries.

Pass any store to the retriever:

```python
retriever = RagWikiRetriever(
    user_id          = "user-1",
    global_retriever = your_retriever,
    state_store      = store,   # swap freely
)
```

---

## Decay Scheduler

Run the decay engine on a background schedule so document states stay current:

```python
from rag_wiki import DecayEngine, DecayConfig, MemoryStateStore
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.scheduler import DecayScheduler

store     = MemoryStateStore()
engine    = DecayEngine(store, StateMachine(), config=DecayConfig())
scheduler = DecayScheduler(engine, store, backend="simple", interval_hours=24)

scheduler.start()

# Manual triggers:
scheduler.run_now("user-123")   # single user
scheduler.run_all_users()       # all users with CLAIMED or PINNED docs

scheduler.stop()
```

Use `backend="apscheduler"` for production (requires `pip install 'langchain-rag-wiki[scheduler]'`).

---

## LlamaIndex Adapter

Wrap any LlamaIndex retriever as a global backend:

```python
from llama_index.core import VectorStoreIndex
from rag_wiki.adapters.llamaindex import LlamaIndexRetrieverAdapter
from rag_wiki import RagWikiRetriever

li_retriever = index.as_retriever(similarity_top_k=5)
adapter      = LlamaIndexRetrieverAdapter(llama_retriever=li_retriever)

retriever = RagWikiRetriever(
    user_id          = "user-1",
    global_retriever = adapter,
)
```

Requires `pip install 'langchain-rag-wiki[llama]'`.

---

## Running Tests

```bash
pip install 'langchain-rag-wiki[dev]'
pytest tests/ -v
```

---

## Project Structure

```
rag_wiki/
├── __init__.py              # public exports
├── retriever.py             # RagWikiRetriever — main entry point
├── scheduler.py             # DecayScheduler (simple + APScheduler backends)
├── storage/
│   ├── base.py              # StateStore ABC + UserDocRecord + DocumentState
│   ├── memory.py            # MemoryStateStore (default, zero deps)
│   ├── sqlite.py            # SQLiteStateStore
│   └── redis_store.py       # RedisStateStore
├── lifecycle/
│   ├── state_machine.py     # pure transition logic
│   ├── fetch_counter.py     # threshold tracking + suggestion events
│   └── decay_engine.py      # scoring + pin/demotion transitions
├── transparency/
│   └── provenance.py        # ProvenanceBlock + ProvenanceBuilder
└── adapters/
    └── llamaindex.py        # LlamaIndex → LangChain adapter
```

---

## Contributing

Contributions are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## License

[MIT](LICENSE)
