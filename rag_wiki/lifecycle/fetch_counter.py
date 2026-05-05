"""
FetchCounter — increments per-user fetch counts and auto-saves documents.

Auto-save (fully automated lifecycle):
  - When fetch_count reaches fetch_threshold, the document is automatically
    transitioned to CLAIMED — no user interaction needed.
  - An AutoSaveEvent is emitted so callers can log / notify.
  - The save-delete lifecycle is now fully automated:
    SURFACED → CLAIMED (auto) → PINNED/DEMOTED (via decay)

Threshold reset:
  - Every query, docs that were NOT retrieved have their queries_missed counter
    incremented by the retriever.
  - When queries_missed >= reset_threshold, fetch_count resets to 0,
    giving the doc a fresh start.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.lifecycle.state_machine import StateMachine

logger = logging.getLogger(__name__)


@dataclass
class AutoSaveEvent:
    """Emitted when a document is auto-saved to the user's personal KB."""
    user_id:     str
    doc_id:      str
    doc_title:   str
    doc_path:    str
    fetch_count: int


# Backwards-compatible alias
SuggestionEvent = AutoSaveEvent


class FetchCounter:
    """
    Tracks how many times each user has retrieved each document,
    and auto-saves (transitions to CLAIMED) when the threshold is crossed.
    """

    def __init__(
        self,
        store:               StateStore,
        state_machine:       StateMachine,
        fetch_threshold:     int = 3,
        no_resiluggest_days: int = 30,   # kept for API compat, unused
        reset_threshold:     int = 3,    # queries without a hit before fetch_count resets
    ):
        self._store           = store
        self._sm              = state_machine
        self._threshold       = fetch_threshold
        self._reset_threshold = reset_threshold

    # ─── Public API ────────────────────────────────────────────────────────────

    def record_fetch(
        self,
        user_id:   str,
        doc_id:    str,
        doc_title: str,
        doc_path:  str,
        now:       Optional[datetime] = None,
    ) -> Optional[AutoSaveEvent]:
        """
        Call every time a document is retrieved for a user.
        Returns an AutoSaveEvent if this fetch crosses the threshold and the
        document was auto-saved, otherwise None.
        """
        now = now or datetime.now(timezone.utc)
        record = self._get_or_create(user_id, doc_id, doc_title, doc_path, now)

        # Pinned/claimed docs use a different retrieval path — don't count
        if record.user_state in (DocumentState.CLAIMED, DocumentState.PINNED):
            return None

        # Doc was seen again — reset missed-query counter
        record.queries_missed  = 0
        record.fetch_count    += 1
        record.last_fetched_at = now

        event = self._check_auto_save(record, now)
        self._store.upsert(record)
        return event

    def record_miss(self, user_id: str, doc_id: str) -> None:
        """
        Call for every tracked SURFACED doc that was NOT retrieved
        in a given query. When queries_missed reaches reset_threshold, the
        fetch_count is reset.
        """
        record = self._store.get(user_id, doc_id)
        if record is None:
            return
        if record.user_state in (DocumentState.CLAIMED, DocumentState.PINNED):
            return

        record.queries_missed += 1

        if record.queries_missed >= self._reset_threshold:
            record.fetch_count      = 0
            record.queries_missed   = 0
            record.user_state       = DocumentState.SURFACED
            logger.debug(
                "Threshold reset for doc_id=%r user_id=%r "
                "(not seen for %d queries)",
                doc_id, user_id, self._reset_threshold,
            )

        self._store.upsert(record)

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
            record.fetch_count     = 0
            self._store.upsert(record)
        return record

    def _check_auto_save(
        self,
        record: UserDocRecord,
        now:    datetime,
    ) -> Optional[AutoSaveEvent]:
        """Auto-save the document if fetch_count has reached the threshold."""
        if record.user_state not in (DocumentState.SURFACED, DocumentState.SUGGESTED):
            return None

        if record.fetch_count < self._threshold:
            return None

        # Auto-transition to CLAIMED
        record = self._sm.transition(record, DocumentState.CLAIMED, now=now)
        logger.info(
            "Auto-saved doc_id=%r for user_id=%r (fetch_count=%d)",
            record.doc_id, record.user_id, record.fetch_count,
        )

        return AutoSaveEvent(
            user_id     = record.user_id,
            doc_id      = record.doc_id,
            doc_title   = record.doc_title,
            doc_path    = record.doc_path,
            fetch_count = record.fetch_count,
        )
