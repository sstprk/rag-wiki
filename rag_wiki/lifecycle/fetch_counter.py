"""
FetchCounter — increments per-user fetch counts and fires save suggestions
when the threshold is met. Coordinates with StateMachine for state updates.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.lifecycle.state_machine import StateMachine


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise a datetime to UTC-aware. Naive datetimes are assumed UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class SuggestionEvent:
    """Emitted when a document crosses the fetch threshold."""
    user_id:   str
    doc_id:    str
    doc_title: str
    doc_path:  str
    fetch_count: int


class FetchCounter:
    """
    Tracks how many times each user has retrieved each document,
    and emits a SuggestionEvent when the threshold is crossed.

    The suggestion fires:
      - exactly once per document (suggestion_sent flag)
      - only if the doc is in SURFACED state
      - only if no_resiluggest_until has passed (after a prior decline)
    """

    def __init__(
        self,
        store:           StateStore,
        state_machine:   StateMachine,
        fetch_threshold: int = 3,
        no_resiluggest_days: int = 30,
    ):
        self._store             = store
        self._sm                = state_machine
        self._threshold         = fetch_threshold
        self._no_resiluggest_days = no_resiluggest_days

    # ─── Public API ────────────────────────────────────────────────────────────

    def record_fetch(
        self,
        user_id:   str,
        doc_id:    str,
        doc_title: str,
        doc_path:  str,
        now:       Optional[datetime] = None,
    ) -> Optional[SuggestionEvent]:
        """
        Call this every time a document is retrieved for a user.

        Returns a SuggestionEvent if this fetch crosses the threshold
        and a suggestion should be shown, otherwise None.
        """
        now = now or datetime.now(timezone.utc)
        record = self._get_or_create(user_id, doc_id, doc_title, doc_path, now)

        # Don't double-count pinned/claimed — they use a different retrieval path
        if record.user_state in (DocumentState.CLAIMED, DocumentState.PINNED):
            return None

        # Increment
        record.fetch_count    += 1
        record.last_fetched_at = now

        suggestion = self._check_suggestion(record, now)
        self._store.upsert(record)
        return suggestion

    def accept_suggestion(self, user_id: str, doc_id: str, full_content: str) -> UserDocRecord:
        """
        User clicked 'Save to my KB'. Transition to CLAIMED and store content.
        """
        record = self._store.get(user_id, doc_id)
        if record is None:
            raise ValueError(f"No record for user={user_id}, doc={doc_id}")

        record = self._sm.transition(record, DocumentState.CLAIMED)
        record.full_content = full_content
        self._store.upsert(record)
        return record

    def decline_suggestion(
        self,
        user_id: str,
        doc_id:  str,
        now:     Optional[datetime] = None,
    ) -> UserDocRecord:
        """
        User clicked 'Not now'. Stay in SURFACED but block re-suggestion
        for no_resiluggest_days.
        """
        now = now or datetime.now(timezone.utc)
        record = self._store.get(user_id, doc_id)
        if record is None:
            raise ValueError(f"No record for user={user_id}, doc={doc_id}")

        record.user_state           = DocumentState.SURFACED
        record.suggestion_sent      = False   # reset so cycle can restart later
        record.no_resiluggest_until = now + timedelta(days=self._no_resiluggest_days)
        self._store.upsert(record)
        return record

    # ─── Private ───────────────────────────────────────────────────────────────

    def _get_or_create(
        self,
        user_id:   str,
        doc_id:    str,
        doc_title: str,
        doc_path:  str,
        now:       datetime,
    ) -> UserDocRecord:
        record = self._store.get(user_id, doc_id)
        if record is None:
            record = self._sm.make_surfaced(user_id, doc_id, doc_title, doc_path)
            record.last_fetched_at = now
            record.fetch_count     = 0   # will be incremented by caller
            self._store.upsert(record)
        return record

    def _check_suggestion(
        self,
        record: UserDocRecord,
        now:    datetime,
    ) -> Optional[SuggestionEvent]:
        """Return a SuggestionEvent if conditions are met, else None."""
        if record.fetch_count < self._threshold:
            return None
        if record.suggestion_sent:
            return None
        if record.user_state != DocumentState.SURFACED:
            return None
        if (
            record.no_resiluggest_until is not None
            and now < _ensure_utc(record.no_resiluggest_until)
        ):
            return None

        # Mark as suggested
        record.user_state     = DocumentState.SUGGESTED
        record.suggestion_sent = True

        return SuggestionEvent(
            user_id     = record.user_id,
            doc_id      = record.doc_id,
            doc_title   = record.doc_title,
            doc_path    = record.doc_path,
            fetch_count = record.fetch_count,
        )