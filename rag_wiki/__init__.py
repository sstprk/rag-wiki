"""
rag-wiki — LangChain-compatible hybrid knowledge base retrieval.

Combines global RAG retrieval with a personal, user-curated knowledge layer
that automatically saves frequently-accessed documents.
"""

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.storage.memory import MemoryStateStore
from rag_wiki.retriever import RagWikiRetriever, RagWikiRetrieverConfig
from rag_wiki.lifecycle.state_machine import StateMachine
from rag_wiki.lifecycle.fetch_counter import FetchCounter, AutoSaveEvent, SuggestionEvent
from rag_wiki.lifecycle.decay_engine import DecayEngine, DecayConfig, DecayResult
from rag_wiki.transparency.provenance import (
    ProvenanceBlock,
    ProvenanceBuilder,
    SourceEntry,
)

__all__ = [
    # Core retriever
    "RagWikiRetriever",
    "RagWikiRetrieverConfig",
    # Storage
    "StateStore",
    "MemoryStateStore",
    "UserDocRecord",
    "DocumentState",
    # Lifecycle
    "StateMachine",
    "FetchCounter",
    "AutoSaveEvent",
    "SuggestionEvent",  # backwards compat alias
    "DecayEngine",
    "DecayConfig",
    "DecayResult",
    # Transparency
    "ProvenanceBlock",
    "ProvenanceBuilder",
    "SourceEntry",
]
