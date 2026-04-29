<div align="center">

# 🧠 rag-wiki

**A hybrid RAG architecture that builds personal knowledge bases through usage.**

*Combines vector retrieval, document provenance, and adaptive user curation — so your knowledge base gets smarter the more you use it.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status: RFC / Spec](https://img.shields.io/badge/Status-RFC%20%2F%20Spec-yellow.svg)]()
[![Integrations: LangChain · VS Code](https://img.shields.io/badge/Integrations-LangChain%20%C2%B7%20VS%20Code-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## The Problem with Standard RAG

Standard RAG pipelines retrieve document chunks based on vector similarity. This works well at scale, but has a fundamental limitation: **every query is stateless**. The system has no memory of what was relevant before, no sense of which documents a specific user relies on, and no way to prefer a known-good source over a probabilistically similar one.

The result is a retrieval system that is:

- **Noisy** — chunks from unrelated documents can outscore the right one
- **Opaque** — the user has no visibility into where the context came from
- **Flat** — all documents are treated equally, regardless of proven relevance
- **Amnesiac** — patterns of use are never learned or acted on

---

## The rag-wiki Approach

rag-wiki layers three things on top of standard RAG:

### 1. Origin Metadata at Split Time
Every chunk in the vector database carries rich metadata back to its source document — `doc_id`, `doc_title`, `section`, `domain_tags`, timestamps. This metadata is the foundation of the entire system. It turns isolated chunks back into traceable artifacts.

### 2. A Personal Document Cache
When a document surfaces repeatedly for a user, the system promotes it: the full origin document is fetched and stored in the user's personal cache. Future queries check this cache **first** — deterministically, by semantic match — and only fall back to the global vector search on a miss. A user's known-relevant documents are never at risk of being outscored by noise.

### 3. Wiki-Style Provenance + Adaptive Feedback Loop
Every response that uses RAG context shows the user exactly where it came from. Hit counters track retrieval frequency per user per document. When a document crosses a threshold, the user is prompted — once, non-intrusively — to save it. From there, a decay and promotion mechanism continuously re-ranks each document's priority based on actual usage. Documents that prove consistently relevant get pinned. Documents that stop being retrieved decay gracefully out of the active layer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        INCOMING QUERY                       │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 2 — USER LOCAL DOCUMENT CACHE             │
│                                                             │
│   Semantic match against user's claimed / pinned files      │
│   ✓ HIT  → inject full document, skip global search         │
│   ✗ MISS → fall through to Tier 1                           │
└───────────────────────────────┬─────────────────────────────┘
                                │ (miss only)
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 1 — GLOBAL RAG LAYER                      │
│                                                             │
│   Vector similarity search → chunk + origin metadata        │
│   Increment hit_count[doc_id] for this user                 │
│   Check if hit_count >= SURFACE_THRESHOLD                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│           TIER 3 — TRANSPARENCY + CURATION LAYER            │
│                                                             │
│   Surface: source doc name, section, hit count              │
│   On threshold: "This file came up 3× — save to library?"   │
│   On save: fetch full doc → store in Tier 2 cache           │
└─────────────────────────────────────────────────────────────┘
```

---

## Document State Machine

Each document progresses through states based on retrieval frequency and user action:

```
  [Global-Only]
       │
       │ first retrieval
       ▼
  [Surfaced] ◄─────────────── (user skips, counter resets)
       │
       │ hit_count >= SURFACE_THRESHOLD
       ▼
  [Candidate] ──── user skips ────► (back to Surfaced, reset)
       │
       │ user saves
       ▼
  [Claimed]
       │                    │
       │ consistent usage   │ inactivity > DECAY_WINDOW
       ▼                    ▼
 [Domain-Pinned]        [Archived]
       │                    │
       │ inactivity         │ re-queried
       ▼                    ▼
   [Claimed]            [Surfaced]
```

| State | What it means | Retrieval behavior |
|---|---|---|
| **Global-Only** | Never retrieved by this user | Standard vector search |
| **Surfaced** | Retrieved ≥ 1×, below threshold | Standard vector search + counter |
| **Candidate** | Threshold reached, prompt shown | Standard vector search |
| **Claimed** | User saved to personal cache | Local cache checked first |
| **Domain-Pinned** | Consistently relevant, auto-injected | Always injected for matching domain |
| **Archived** | Was claimed, fell out of use | Removed from priority path; restarts on re-query |

---

## Threshold & Promotion Logic

The save prompt fires **once per threshold crossing** — not on every query. This keeps the UX clean.

```
Default: SURFACE_THRESHOLD = 3
```

- Multiple chunks from the same document in a single query count as **one** increment (no inflation from dense documents)
- If the user skips, the counter resets. The document will surface again after another full threshold cycle.
- The prompt is non-blocking — it appears after the response, never before

**Example prompt:**

```
📂 "API Rate Limiting — Reference Guide" has come up 3 times.
   Save it to your personal library for faster access?

   [Save to Library]   [Not now]
```

---

## Decay & Feedback Loop

Claimed documents that stop being retrieved begin to decay, preventing the personal cache from becoming stale.

```
decay_score = (days_since_last_retrieved / DECAY_WINDOW) × base_weight
```

When `decay_score` exceeds `ARCHIVE_THRESHOLD`, the document moves to **Archived** — not deleted, just removed from the priority retrieval path.

For documents that prove consistently relevant across many sessions, **Domain Pinning** promotes them to automatic context injection:

- Claimed for at least `MIN_CLAIMED_DAYS` (default: 7)
- Retrieved across at least `DOMAIN_PIN_SESSIONS` distinct sessions (default: 5)
- Average relevance score above `DOMAIN_PIN_SCORE_FLOOR` (default: 0.75)

The combined effect: a living personal knowledge base that self-organizes around actual usage patterns rather than one-time decisions.

---

## Planned Integrations

### 🔗 LangChain Plugin
A custom `HybridCacheRetriever` that wraps any standard LangChain retriever and adds the caching, hit counting, and promotion logic as a composable layer. Drop it into any existing LangChain chain without changing the rest of your pipeline.

```python
# Planned interface (spec — not yet implemented)
from rag_wiki.langchain import HybridCacheRetriever

retriever = HybridCacheRetriever(
    base_retriever=your_existing_vectorstore_retriever,
    user_id="user_123",
    cache_store=LocalDocumentCache("./user_cache"),
    surface_threshold=3,
)

# Use it like any other LangChain retriever
docs = retriever.get_relevant_documents("your query")
```

### 🖥️ VS Code Extension
An IDE-level integration that surfaces the provenance layer inline while you work. When the AI assistant retrieves context from your knowledge base, the extension shows a source badge in the editor. Repeated retrievals trigger the save prompt directly in the IDE sidebar — no context switching.

**Planned UX:**
- Source badge on AI suggestions: `📄 contracts/sla-template.md · retrieved 2×`
- Sidebar panel: personal document library with state indicators
- Command palette: `RAG Wiki: View my knowledge base`, `RAG Wiki: Pin document`

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `SURFACE_THRESHOLD` | `3` | Retrievals before save prompt |
| `DECAY_WINDOW` | `30 days` | Inactivity window before decay |
| `ARCHIVE_THRESHOLD` | `0.8` | Decay score that triggers archive |
| `DOMAIN_PIN_SESSIONS` | `5` | Sessions needed for domain pinning |
| `MIN_CLAIMED_DAYS` | `7` | Min days claimed before pin eligible |
| `DOMAIN_PIN_SCORE_FLOOR` | `0.75` | Min avg relevance for pinning |
| `COUNTER_RESET_ON_SKIP` | `true` | Reset counter when user skips prompt |
| `MULTI_CHUNK_COUNT_ONCE` | `true` | Multiple chunks from same doc = 1 hit |

---

## Why This Isn't Just Standard RAG with a Cache

A few things that make this architecture distinct from simply caching retrieval results:

- **The cache stores full documents, not chunks.** Once a document is claimed, future queries get the full coherent source — not fragments. This dramatically reduces chunk-blending hallucinations.
- **The promotion trigger is behavioral, not manual.** Users don't manage a knowledge base explicitly. The system learns from query patterns and only asks for confirmation when it has strong evidence of relevance.
- **The decay loop is continuous.** The personal cache is not a static store. Documents that lose relevance are automatically deprioritized without the user having to curate anything.
- **Provenance is first-class.** Every retrieval is traceable. The user always knows which document answered their question and how often it has been relied on.

---

## Repository Structure

```
rag-wiki/
├── README.md
├── CONTRIBUTING.md
├── docs/
│   ├── architecture.md        ← Full system spec
│   ├── retrieval-flow.md      ← End-to-end query lifecycle
│   ├── document-states.md     ← State machine detail
│   ├── decay-feedback.md      ← Decay formula + pinning logic
│   └── configuration.md      ← All config parameters
├── integrations/
│   ├── langchain/
│   │   └── README.md          ← LangChain plugin interface spec
│   └── vscode/
│       └── README.md          ← VS Code extension UX spec
└── rfcs/
    └── 0001-initial-design.md ← Formal RFC
```

```python
import redis
from rag_wiki.storage.redis_store import RedisStateStore

## Status

This repository is currently a **design spec and RFC**. No implementation code exists yet.

The goal is to:
1. Finalize the architecture spec through community feedback
2. Build the LangChain plugin (`HybridCacheRetriever`) as the first implementation
3. Build the VS Code extension as the second integration target
4. Publish as composable, framework-agnostic packages

---

## Contributing

This project is in the spec phase — the best contributions right now are **feedback, questions, and challenges to the design**.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Some areas actively looking for input:

- Edge cases in the decay / promotion logic
- Alternative threshold strategies (time-based vs count-based)
- Multi-user / team caching scenarios
- Storage backend options for the local document cache
- LangChain retriever interface design

Open an issue or start a discussion. All are welcome.

---

## Author

**[sstprk](https://github.com/sstprk)**

---

<div align="center">

*If this architecture solves a problem you've hit with RAG — open an issue, leave a star, or reach out.*

</div>
