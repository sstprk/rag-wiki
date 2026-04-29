# langchain-rag-wiki

A LangChain-compatible retrieval package that adds a **personal, user-curated knowledge layer** on top of any existing RAG retriever. Documents that a user repeatedly retrieves are surfaced as save suggestions; once accepted, they're served directly from a local cache — skipping vector search entirely — with full provenance transparency after every query.

---

## Installation

```bash
pip install langchain-rag-wiki
```

With optional backends:

```bash
pip install 'langchain-rag-wiki[sqlite]'    # SQLAlchemy-backed store
pip install 'langchain-rag-wiki[redis]'      # Redis-backed store
pip install 'langchain-rag-wiki[scheduler]'  # APScheduler for decay jobs
pip install 'langchain-rag-wiki[llama]'      # LlamaIndex adapter
pip install 'langchain-rag-wiki[dev]'        # pytest + fakeredis + all above
```

---

## Quickstart

```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig, MemoryStateStore

# Wrap any existing LangChain retriever:
retriever = RagWikiRetriever(
    user_id="user-123",
    global_retriever=your_existing_retriever,
    state_store=MemoryStateStore(),               # zero deps, in-memory
    config=RagWikiRetrieverConfig(fetch_threshold=3),
)

docs = retriever.invoke("quarterly earnings")
print(retriever.last_provenance.render())         # see what was used
```

---

## Connecting to a Real Vector DB

### Chroma

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)
global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = RagWikiRetriever(
    user_id="user-1",
    global_retriever=global_retriever,
)
```

### Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=OpenAIEmbeddings(),
)
global_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = RagWikiRetriever(
    user_id="user-1",
    global_retriever=global_retriever,
)
```

---

## Configuration

### RagWikiRetrieverConfig

| Parameter             | Type  | Default | Description                                         |
|-----------------------|-------|---------|-----------------------------------------------------|
| `fetch_threshold`     | `int` | `3`     | Fetches before a save suggestion fires               |
| `no_resiluggest_days` | `int` | `30`    | Days to wait before re-suggesting a declined doc     |
| `decay`               | `DecayConfig` | *(see below)* | Decay engine settings              |

### DecayConfig

| Parameter              | Type    | Default | Description                                      |
|------------------------|---------|---------|--------------------------------------------------|
| `w_recency`            | `float` | `0.5`   | Weight for recency factor in decay score         |
| `w_frequency`          | `float` | `0.3`   | Weight for frequency factor                      |
| `w_explicit`           | `float` | `0.2`   | Weight for explicit user signals                 |
| `decay_lambda`         | `float` | `0.05`  | Exponential decay steepness (λ)                  |
| `freq_cap`             | `int`   | `20`    | Max fetch count for frequency normalization      |
| `pin_threshold`        | `float` | `0.85`  | Score above which a doc is auto-pinned           |
| `demotion_threshold`   | `float` | `0.15`  | Score below which a doc is auto-demoted          |
| `pin_hold_days`        | `int`   | `7`     | Days score must hold before pin fires            |
| `demotion_hold_days`   | `int`   | `3`     | Days score must hold before demotion fires       |

---

## Running the Decay Scheduler

```python
from rag_wiki import DecayEngine, DecayConfig
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.scheduler import DecayScheduler
from rag_wiki.storage.memory import MemoryStateStore

store  = MemoryStateStore()
engine = DecayEngine(store, StateMachine(), config=DecayConfig())

# Simple backend (zero extra deps):
scheduler = DecayScheduler(engine, store, backend="simple", interval_hours=24)
scheduler.start()

# Or with APScheduler:
# scheduler = DecayScheduler(engine, store, backend="apscheduler", interval_hours=24)

# Manual triggers:
scheduler.run_now("user-123")
scheduler.run_all_users()

scheduler.stop()
```

---

## Swapping Storage Backends

### Memory (default, zero deps)

```python
from rag_wiki.storage.memory import MemoryStateStore
store = MemoryStateStore()
```

### SQLite (persistent, single-file)

```python
from rag_wiki.storage.sqlite import SQLiteStateStore
store = SQLiteStateStore("sqlite:///./rag_wiki.db")
```

### Redis (distributed, production)

```python
import redis
from rag_wiki.storage.redis_store import RedisStateStore

client = redis.Redis(host="localhost", port=6379, db=0)
store = RedisStateStore(client)
```

Pass any store to `RagWikiRetriever(state_store=store)`.

---

## Retrieval Priority Order

1. **PINNED** — always injected into context
2. **CLAIMED** — served from local cache (skips vector search)
3. **Global RAG** — fallback vector similarity search

---

## Running Tests

```bash
pip install 'langchain-rag-wiki[dev]'
pytest tests/ -v
```

---

## License

MIT
