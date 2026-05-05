"""
PineconeRetriever — A LangChain BaseRetriever backed by the native Pinecone SDK.

Since langchain-pinecone doesn't support Python 3.14, this adapter wraps the
Pinecone SDK (v9) directly as a LangChain-compatible retriever that works
seamlessly with RagWikiRetriever.

Usage:
    from rag_wiki.adapters.pinecone_retriever import PineconeRetriever

    retriever = PineconeRetriever(
        api_key    = "pcsk_...",
        index_name = "my-index",
        embed_model = OllamaEmbeddings(model="mxbai-embed-large:latest"),
        top_k      = 5,
    )
    docs = retriever.invoke("my query")
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class PineconeRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that queries a Pinecone index directly
    using the native Pinecone SDK.

    The embedding model is exposed via the ``embeddings`` attribute so
    RagWikiRetriever can auto-resolve it for semantic cache lookups.
    """

    embeddings:  Any   # LangChain embedding model (exposed for auto-resolution)
    top_k:       int  = 5
    namespace:   Optional[str] = None

    # Internal — set via object.__setattr__
    _index: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        api_key:     str,
        index_name:  str,
        embed_model: Any,
        top_k:       int = 5,
        namespace:   Optional[str] = None,
        **kwargs,
    ):
        from pinecone import Pinecone

        pc    = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        super().__init__(
            embeddings = embed_model,
            top_k      = top_k,
            namespace  = namespace,
            **kwargs,
        )
        object.__setattr__(self, "_index", index)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Embed the query, search Pinecone, return Documents with metadata."""
        index = object.__getattribute__(self, "_index")

        # Embed
        query_vec = self.embeddings.embed_query(query)

        # Query Pinecone
        results = index.query(
            vector=query_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace or None,
        )

        docs = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            text = meta.pop("text", "")
            # Convert numeric metadata back to int
            if "chunk_index" in meta:
                meta["chunk_index"] = int(meta["chunk_index"])
            if "total_chunks" in meta:
                meta["total_chunks"] = int(meta["total_chunks"])
            meta["score"] = match.get("score", 0.0)
            docs.append(Document(page_content=text, metadata=meta))

        return docs
