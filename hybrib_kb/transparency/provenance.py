"""
ProvenanceBlock — builds the transparency layer shown to users after each query.
Returns both a structured dict (for programmatic use) and a formatted string
(for direct display in chat/UI).
"""

from dataclasses import dataclass, field
from typing import Optional

from hybrid_kb.storage.base import DocumentState, UserDocRecord
from hybrid_kb.lifecycle.fetch_counter import SuggestionEvent


@dataclass
class SourceEntry:
    doc_id:          str
    doc_title:       str
    doc_path:        str
    user_state:      DocumentState
    fetch_count:     int
    chunks_used:     list[int]      = field(default_factory=list)
    total_chunks:    Optional[int]  = None
    section_heading: Optional[str]  = None
    from_cache:      bool           = False   # True if served from local cache


@dataclass
class ProvenanceBlock:
    sources:    list[SourceEntry]
    suggestion: Optional[SuggestionEvent] = None

    # ─── Text rendering ────────────────────────────────────────────────────────

    def render(self) -> str:
        """
        Returns a human-readable provenance block suitable for appending
        to any chat response.
        """
        lines = ["", "─" * 60, "📄 Sources used in this response"]

        for src in self.sources:
            state_label = self._state_label(src)
            chunk_info  = self._chunk_info(src)
            cache_badge = " [from your KB]" if src.from_cache else ""
            lines.append(
                f"  • {src.doc_title}{cache_badge}"
                f"\n    {chunk_info}  |  {state_label}"
            )
            if src.section_heading:
                lines.append(f"    Section: {src.section_heading}")

        if self.suggestion:
            lines += self._render_suggestion(self.suggestion)

        lines.append("─" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Structured representation for API consumers."""
        return {
            "sources": [
                {
                    "doc_id":          s.doc_id,
                    "doc_title":       s.doc_title,
                    "doc_path":        s.doc_path,
                    "user_state":      s.user_state.value,
                    "fetch_count":     s.fetch_count,
                    "chunks_used":     s.chunks_used,
                    "total_chunks":    s.total_chunks,
                    "section_heading": s.section_heading,
                    "from_cache":      s.from_cache,
                }
                for s in self.sources
            ],
            "suggestion": {
                "doc_id":    self.suggestion.doc_id,
                "doc_title": self.suggestion.doc_title,
                "fetch_count": self.suggestion.fetch_count,
            } if self.suggestion else None,
        }

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _state_label(self, src: SourceEntry) -> str:
        labels = {
            DocumentState.SURFACED:  f"SURFACED (fetched {src.fetch_count}×)",
            DocumentState.SUGGESTED: f"SUGGESTED (fetched {src.fetch_count}×)",
            DocumentState.CLAIMED:   "Saved to your KB",
            DocumentState.PINNED:    "📌 Pinned to your KB",
            DocumentState.DEMOTED:   "DEMOTED",
            DocumentState.GLOBAL:    "GLOBAL",
        }
        return labels.get(src.user_state, src.user_state.value)

    def _chunk_info(self, src: SourceEntry) -> str:
        if not src.chunks_used:
            return "Full document"
        chunks = ", ".join(str(c) for c in src.chunks_used)
        total  = f" of {src.total_chunks}" if src.total_chunks else ""
        return f"Chunks {chunks}{total}"

    def _render_suggestion(self, s: SuggestionEvent) -> list[str]:
        return [
            "",
            f"💡 \"{s.doc_title}\" has appeared in your queries {s.fetch_count} times.",
            "   Would you like to save it to your personal knowledge base",
            "   for faster, direct access?",
            "",
            "   [ Save to my KB ]   [ Not now ]",
            "   → Call retriever.accept_suggestion() or retriever.decline_suggestion()",
        ]


class ProvenanceBuilder:
    """
    Assembles a ProvenanceBlock from retrieved documents and lifecycle records.
    Designed to be called at the end of HybridRetriever._get_relevant_documents().
    """

    def build(
        self,
        retrieved_docs: list[dict],          # list of dicts with chunk metadata
        user_records:   dict[str, UserDocRecord],  # doc_id → UserDocRecord
        suggestion:     Optional[SuggestionEvent] = None,
    ) -> ProvenanceBlock:
        """
        retrieved_docs: each dict must have at minimum:
            doc_id, doc_title, doc_path, chunk_index, from_cache
            optionally: total_chunks, section_heading
        user_records: mapping from doc_id to its current UserDocRecord
        """
        # Group chunks by doc_id
        doc_chunks: dict[str, list[dict]] = {}
        for doc in retrieved_docs:
            doc_id = doc.get("doc_id", "unknown")
            doc_chunks.setdefault(doc_id, []).append(doc)

        sources = []
        for doc_id, chunks in doc_chunks.items():
            first    = chunks[0]
            record   = user_records.get(doc_id)
            sources.append(SourceEntry(
                doc_id          = doc_id,
                doc_title       = first.get("doc_title", "Unknown Document"),
                doc_path        = first.get("doc_path", ""),
                user_state      = record.user_state if record else DocumentState.GLOBAL,
                fetch_count     = record.fetch_count if record else 0,
                chunks_used     = [c.get("chunk_index") for c in chunks
                                   if c.get("chunk_index") is not None],
                total_chunks    = first.get("total_chunks"),
                section_heading = first.get("section_heading"),
                from_cache      = first.get("from_cache", False),
            ))

        return ProvenanceBlock(sources=sources, suggestion=suggestion)