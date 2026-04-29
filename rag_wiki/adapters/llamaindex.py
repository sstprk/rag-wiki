"""
LlamaIndexRetrieverAdapter — wraps any LlamaIndex retriever as a LangChain
:class:`BaseRetriever`, so it can be passed as ``global_retriever`` to
:class:`RagWikiRetriever`.

.. note::
   Requires ``llama-index-core`` (``pip install llama-index-core>=0.10``).
   If ``llama_index`` is not installed, importing this module raises an
   ``ImportError`` with a helpful message.

Usage::

    from llama_index.core import VectorStoreIndex
    from rag_wiki.adapters.llamaindex import LlamaIndexRetrieverAdapter

    li_retriever = index.as_retriever(similarity_top_k=5)
    adapter      = LlamaIndexRetrieverAdapter(llama_retriever=li_retriever)

    # Now use it with RagWikiRetriever:
    hybrid = RagWikiRetriever(
        user_id="user-1",
        global_retriever=adapter,
    )
"""

from __future__ import annotations

from typing import Any, List

try:
    from llama_index.core.base.base_retriever import BaseRetriever as LIBaseRetriever
except ImportError as exc:
    raise ImportError(
        "LlamaIndexRetrieverAdapter requires the 'llama-index-core' package. "
        "Install it with:  pip install 'langchain-rag-wiki[llama]'"
    ) from exc

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class LlamaIndexRetrieverAdapter(BaseRetriever):
    """
    Adapts a LlamaIndex retriever to the LangChain :class:`BaseRetriever`
    interface.

    Maps :class:`NodeWithScore` objects to LangChain :class:`Document`
    objects, preserving metadata including ``doc_id``, ``doc_title``,
    ``source``, and ``score``.

    Parameters
    ----------
    llama_retriever:
        A ``llama_index.core.base.BaseRetriever`` instance.
    """

    llama_retriever: Any  # LIBaseRetriever — typed as Any for Pydantic

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Query the wrapped LlamaIndex retriever and convert results to
        LangChain Documents.
        """
        nodes = self.llama_retriever.retrieve(query)
        docs: List[Document] = []

        for node_with_score in nodes:
            node = node_with_score.node
            metadata = dict(node.metadata) if node.metadata else {}

            # Preserve score from LlamaIndex
            metadata["score"] = node_with_score.score

            # Ensure standard rag-wiki metadata keys are present
            if "doc_id" not in metadata and hasattr(node, "node_id"):
                metadata["doc_id"] = node.node_id
            if "doc_title" not in metadata:
                metadata["doc_title"] = metadata.get("title", metadata.get("source", "Unknown"))
            if "source" not in metadata:
                metadata["source"] = metadata.get("file_path", metadata.get("doc_path", ""))

            docs.append(
                Document(
                    page_content=node.get_content(),
                    metadata=metadata,
                )
            )

        return docs
