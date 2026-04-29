"""
StateMachine — pure logic, no I/O.
Defines valid transitions and applies them to UserDocRecord instances.
"""

from datetime import datetime
from typing import Optional

from hybrid_kb.storage.base import DocumentState, UserDocRecord


# Valid transitions: state → set of states it can move to
TRANSITIONS: dict[DocumentState, list[DocumentState]] = {
    DocumentState.GLOBAL:    [DocumentState.SURFACED],
    DocumentState.SURFACED:  [DocumentState.SUGGESTED, DocumentState.DEMOTED],
    DocumentState.SUGGESTED: [DocumentState.CLAIMED, DocumentState.SURFACED],
    DocumentState.CLAIMED:   [DocumentState.PINNED, DocumentState.DEMOTED],
    DocumentState.PINNED:    [DocumentState.CLAIMED, DocumentState.DEMOTED],
    DocumentState.DEMOTED:   [DocumentState.SURFACED],
}


class InvalidTransitionError(Exception):
    pass


class StateMachine:

    def can_transition(
        self,
        record: UserDocRecord,
        to: DocumentState,
    ) -> bool:
        return to in TRANSITIONS.get(record.user_state, [])

    def transition(
        self,
        record: UserDocRecord,
        to: DocumentState,
        now: Optional[datetime] = None,
    ) -> UserDocRecord:
        """
        Apply a state transition and stamp the relevant timestamp.
        Raises InvalidTransitionError if the transition is not allowed.
        """
        if not self.can_transition(record, to):
            raise InvalidTransitionError(
                f"Cannot transition {record.user_state!r} → {to!r} "
                f"for doc_id={record.doc_id!r}"
            )

        now = now or datetime.utcnow()
        record.user_state = to

        if to == DocumentState.CLAIMED:
            record.cached_at  = now
            record.demoted_at = None

        elif to == DocumentState.PINNED:
            record.pinned_at = now

        elif to == DocumentState.DEMOTED:
            record.demoted_at  = now
            record.full_content = None   # evict cached content
            record.pinned_at   = None

        elif to == DocumentState.SURFACED:
            # coming back from DEMOTED or SUGGESTED rejection
            record.demoted_at        = None
            record.suggestion_sent   = False  # allow re-suggestion cycle
            record.no_resiluggest_until = None

        return record

    # ─── Convenience constructors ──────────────────────────────────────────────

    def make_surfaced(
        self,
        user_id:   str,
        doc_id:    str,
        doc_title: str,
        doc_path:  str,
    ) -> UserDocRecord:
        """
        Create a brand-new SURFACED record for a doc the user just encountered
        for the first time via global RAG.
        """
        return UserDocRecord(
            user_id    = user_id,
            doc_id     = doc_id,
            doc_title  = doc_title,
            doc_path   = doc_path,
            user_state = DocumentState.SURFACED,
            fetch_count = 1,
            last_fetched_at = datetime.utcnow(),
        )