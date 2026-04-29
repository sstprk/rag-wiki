"""
Tests for LlamaIndexRetrieverAdapter.
Mocks the LlamaIndex retriever so llama-index-core is not required.
"""

import sys
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document


# ─── Fake llama_index module structure ────────────────────────────────────────
# We build a minimal fake so the adapter's try/import succeeds without
# llama-index-core being installed.


@dataclass
class FakeTextNode:
    """Mimics ``llama_index.core.schema.TextNode``."""

    text: str = ""
    node_id: str = ""
    metadata: dict = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text


@dataclass
class FakeNodeWithScore:
    """Mimics ``llama_index.core.schema.NodeWithScore``."""

    node: FakeTextNode = field(default_factory=FakeTextNode)
    score: Optional[float] = None


class FakeLIBaseRetriever:
    """Mimics ``llama_index.core.base.base_retriever.BaseRetriever``."""

    def retrieve(self, query: str) -> list:
        raise NotImplementedError


def _install_fake_llama_modules() -> None:
    """Inject fake llama_index modules into sys.modules."""
    li = ModuleType("llama_index")
    li_core = ModuleType("llama_index.core")
    li_base = ModuleType("llama_index.core.base")
    li_base_retriever = ModuleType("llama_index.core.base.base_retriever")

    li_base_retriever.BaseRetriever = FakeLIBaseRetriever  # type: ignore[attr-defined]

    sys.modules["llama_index"] = li
    sys.modules["llama_index.core"] = li_core
    sys.modules["llama_index.core.base"] = li_base
    sys.modules["llama_index.core.base.base_retriever"] = li_base_retriever

    li.core = li_core  # type: ignore[attr-defined]
    li_core.base = li_base  # type: ignore[attr-defined]
    li_base.base_retriever = li_base_retriever  # type: ignore[attr-defined]


def _cleanup_fake_llama_modules() -> None:
    """Remove fake llama_index modules from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("llama_index"):
            del sys.modules[key]
    # Also remove the adapter so it gets re-imported fresh
    for key in list(sys.modules.keys()):
        if "llamaindex" in key:
            del sys.modules[key]


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fake_llama():
    """Install fake llama_index modules before each test, clean up after."""
    _install_fake_llama_modules()
    yield
    _cleanup_fake_llama_modules()


def _import_adapter():
    """Import the adapter after fake modules are installed."""
    # Force reimport so the try/import picks up fake modules
    if "rag_wiki.adapters.llamaindex" in sys.modules:
        del sys.modules["rag_wiki.adapters.llamaindex"]
    from rag_wiki.adapters.llamaindex import LlamaIndexRetrieverAdapter
    return LlamaIndexRetrieverAdapter


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestLlamaIndexAdapter:
    def test_empty_retrieval(self) -> None:
        """Adapter returns empty list when LlamaIndex returns no nodes."""
        Adapter = _import_adapter()
        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = []

        adapter = Adapter(llama_retriever=mock_li)
        docs = adapter.invoke("test query")
        assert docs == []
        mock_li.retrieve.assert_called_once()

    def test_converts_nodes_to_documents(self) -> None:
        """Each NodeWithScore becomes a LangChain Document with metadata."""
        Adapter = _import_adapter()
        node1 = FakeTextNode(
            text="Hello world",
            node_id="n1",
            metadata={
                "doc_id": "doc-1",
                "doc_title": "Greeting Doc",
                "source": "/kb/greeting.pdf",
            },
        )
        node2 = FakeTextNode(
            text="Goodbye world",
            node_id="n2",
            metadata={
                "doc_id": "doc-2",
                "title": "Farewell Doc",
            },
        )

        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = [
            FakeNodeWithScore(node=node1, score=0.95),
            FakeNodeWithScore(node=node2, score=0.80),
        ]

        adapter = Adapter(llama_retriever=mock_li)
        docs = adapter.invoke("query")

        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Hello world"
        assert docs[0].metadata["doc_id"] == "doc-1"
        assert docs[0].metadata["doc_title"] == "Greeting Doc"
        assert docs[0].metadata["score"] == 0.95

    def test_preserves_score_in_metadata(self) -> None:
        """The LlamaIndex score should appear in the Document metadata."""
        Adapter = _import_adapter()
        node = FakeTextNode(text="content", node_id="n1", metadata={})
        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = [
            FakeNodeWithScore(node=node, score=0.42),
        ]

        adapter = Adapter(llama_retriever=mock_li)
        docs = adapter.invoke("q")
        assert docs[0].metadata["score"] == 0.42

    def test_fallback_doc_id_from_node_id(self) -> None:
        """When metadata has no doc_id, node_id is used as fallback."""
        Adapter = _import_adapter()
        node = FakeTextNode(text="content", node_id="fallback-id", metadata={})
        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = [
            FakeNodeWithScore(node=node, score=0.5),
        ]

        adapter = Adapter(llama_retriever=mock_li)
        docs = adapter.invoke("q")
        assert docs[0].metadata["doc_id"] == "fallback-id"

    def test_fallback_doc_title_from_source(self) -> None:
        """When metadata has no doc_title or title, source is used."""
        Adapter = _import_adapter()
        node = FakeTextNode(
            text="content",
            node_id="n1",
            metadata={"source": "my_source.pdf"},
        )
        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = [
            FakeNodeWithScore(node=node, score=0.5),
        ]

        adapter = Adapter(llama_retriever=mock_li)
        docs = adapter.invoke("q")
        assert docs[0].metadata["doc_title"] == "my_source.pdf"

    def test_multiple_queries_independent(self) -> None:
        """Each invoke call should produce fresh results."""
        Adapter = _import_adapter()
        node = FakeTextNode(text="hello", node_id="n1", metadata={})
        mock_li = MagicMock(spec=FakeLIBaseRetriever)
        mock_li.retrieve.return_value = [
            FakeNodeWithScore(node=node, score=0.9),
        ]

        adapter = Adapter(llama_retriever=mock_li)
        docs1 = adapter.invoke("q1")
        docs2 = adapter.invoke("q2")

        assert len(docs1) == 1
        assert len(docs2) == 1
        assert mock_li.retrieve.call_count == 2
