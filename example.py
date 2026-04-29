#!/usr/bin/env python3
"""
example.py — Complete runnable demo of rag-wiki.

Uses only MemoryStateStore (no infra needed) and a MockRetriever that
returns fake documents.  No API keys or external services required.

Run with:  python example.py
"""

from __future__ import annotations

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_wiki import (
    RagWikiRetriever,
    RagWikiRetrieverConfig,
    MemoryStateStore,
)
from rag_wiki.lifecycle.fetch_counter import SuggestionEvent


# ─── Mock retriever ───────────────────────────────────────────────────────────


class MockRetriever(BaseRetriever):
    """Returns 3 fake documents with proper metadata."""

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [
            Document(
                page_content=(
                    "Kubernetes uses pods as the smallest deployable unit. "
                    "Each pod runs one or more containers sharing network and storage."
                ),
                metadata={
                    "doc_id": "k8s-pods",
                    "doc_title": "Kubernetes Pod Basics",
                    "doc_path": "/docs/k8s/pods.md",
                    "chunk_index": 0,
                    "total_chunks": 5,
                },
            ),
            Document(
                page_content=(
                    "Docker images are built from Dockerfiles. "
                    "Each instruction creates a layer in the image filesystem."
                ),
                metadata={
                    "doc_id": "docker-images",
                    "doc_title": "Docker Image Guide",
                    "doc_path": "/docs/docker/images.md",
                    "chunk_index": 0,
                    "total_chunks": 3,
                },
            ),
            Document(
                page_content=(
                    "Helm charts package Kubernetes resources into reusable bundles. "
                    "A chart contains templates, values, and metadata."
                ),
                metadata={
                    "doc_id": "helm-charts",
                    "doc_title": "Helm Charts Overview",
                    "doc_path": "/docs/helm/charts.md",
                    "chunk_index": 0,
                    "total_chunks": 4,
                },
            ),
        ]


# ─── Simulation ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full simulation loop."""
    store = MemoryStateStore()
    suggestion_events: list[SuggestionEvent] = []

    def on_suggestion(event: SuggestionEvent) -> None:
        suggestion_events.append(event)
        print(f"\n💡 SUGGESTION FIRED: \"{event.doc_title}\" "
              f"(fetched {event.fetch_count}× for user {event.user_id})")

    retriever = RagWikiRetriever(
        user_id="demo-user",
        global_retriever=MockRetriever(),
        state_store=store,
        config=RagWikiRetrieverConfig(fetch_threshold=2),
        on_suggestion=on_suggestion,
    )

    # ── Phase 1: 5 queries → triggers suggestions ─────────────────────────────
    print("=" * 60)
    print("PHASE 1: 5 queries (fetch_threshold=2, suggestions will fire)")
    print("=" * 60)

    for i in range(1, 6):
        print(f"\n─── Query {i} ───")
        docs = retriever.invoke("kubernetes pods docker")
        print(f"  Retrieved {len(docs)} documents")
        print(retriever.last_provenance.render())

    # ── Accept the first suggestion ───────────────────────────────────────────
    if suggestion_events:
        first = suggestion_events[0]
        print(f"\n✅ ACCEPTING suggestion for \"{first.doc_title}\"")
        retriever.accept_suggestion(
            doc_id=first.doc_id,
            full_content=(
                "Kubernetes uses pods as the smallest deployable unit. "
                "Each pod runs one or more containers sharing network and storage. "
                "Pods are ephemeral — when they crash, the controller creates new ones. "
                "You can group related containers in a single pod for tight coupling."
            ),
        )
        print(f"  → Document is now CLAIMED and cached locally.\n")

    # ── Phase 2: 3 more queries → cached doc served directly ──────────────────
    print("=" * 60)
    print("PHASE 2: 3 more queries (claimed doc served from cache)")
    print("=" * 60)

    for i in range(6, 9):
        print(f"\n─── Query {i} ───")
        docs = retriever.invoke("kubernetes pods")
        cached = [d for d in docs if d.metadata.get("from_cache")]
        print(f"  Retrieved {len(docs)} documents ({len(cached)} from cache)")
        print(retriever.last_provenance.render())

    # ── Manual decay ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Manual decay run")
    print("=" * 60)

    results = retriever.run_decay()
    for r in results:
        print(f"  {r.doc_id}: score {r.old_score:.3f} → {r.new_score:.3f}  "
              f"[{r.old_state.value} → {r.new_state.value}]"
              f"{'  ⚡ TRANSITIONED' if r.transitioned else ''}")

    if not results:
        print("  (no CLAIMED/PINNED docs to decay)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total suggestions fired: {len(suggestion_events)}")
    for evt in suggestion_events:
        print(f"    • {evt.doc_title} (doc_id={evt.doc_id}, "
              f"fetches={evt.fetch_count})")

    # Show final state of all records
    print("\n  Final record states:")
    for doc_id in ["k8s-pods", "docker-images", "helm-charts"]:
        record = store.get("demo-user", doc_id)
        if record:
            print(f"    • {record.doc_title}: {record.user_state.value} "
                  f"(fetches={record.fetch_count}, "
                  f"decay_score={record.decay_score:.3f})")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
