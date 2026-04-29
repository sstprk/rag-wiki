"""
FetchCounter — increments per-user fetch counts and fires save suggestions.

Suggestion escalation:
  - First suggestion fires at fetch_count == fetch_threshold (e.g. 2)
  - After decline, next suggestion fires at fetch_count + (gap * 2)
    e.g. threshold=2 → suggest at 2, then 6, then 14, then 30 ...
  - After accept, no more suggestions (doc is CLAIMED)

Threshold reset:
  - Every query, docs that were NOT retrieved have their queries_missed counter
    incremented by the retriever.
  - When queries_missed >= reset_threshold, fetch_count resets to 0 and
    next_suggest_at resets, giving the doc a fresh start.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord
from rag_wiki.lifecycle.state_machine import StateMachine

logger = logging.getLogger(__name__)


@dataclass
class SuggestionEvent:
    """Emitted when a document crosses the fetch threshold."""
    user_id:     str
    doc_id:      str
    doc_title:   str
    doc_path:    str
    fetch_count: int


class FetchCounter:
    """
    Tracks how many times each user has retrieved each document,
    and emits a SuggestionEvent when the threshold (or escalated target) is crossed.
    """

    def __init__(
        self,
        store:               StateStore,
        state_machine:       StateMachine,
        fetch_threshold:     int = 3,
        no_resiluggest_days: int = 30,   # kept for API compat, no longer used for timing
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
    ) -> Optional[SuggestionEvent]:
        """
        Call every time a document is retrieved for a user.
        Returns a SuggestionEvent if this fetch crosses the current suggestion
        target, otherwise None.
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

        suggestion = self._check_suggestion(record)
        self._store.upsert(record)
        return suggestion

    def record_miss(self, user_id: str, doc_id: str) -> None:
        """
        Call for every tracked SURFACED/SUGGESTED doc that was NOT retrieved
        in a given query. When queries_missed reaches reset_threshold, the
        fetch_count and suggestion target are reset.
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
            record.next_suggest_at  = 0
            record.suggestion_sent  = False
            record.user_state       = DocumentState.SURFACED
            logger.debug(
                "Threshold reset for doc_id=%r user_id=%r "
                "(not seen for %d queries)",
                doc_id, user_id, self._reset_threshold,
            )

        self._store.upsert(record)

    def accept_suggestion(self, user_id: str, doc_id: str, full_content: str) -> UserDocRecord:
        """User clicked 'Save to my KB'. Transition to CLAIMED and store content."""
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
        User clicked 'Not now'. Stay in SURFACED and schedule the next
        suggestion by doubling the gap each time.

        Example with threshold=2:
          1st suggestion at fetch_count=2  → next at 6   (gap doubles: 2→4)
          2nd suggestion at fetch_count=6  → next at 14  (gap doubles: 4→8)
          3rd suggestion at fetch_count=14 → next at 30  (gap doubles: 8→16)
        """
        now = now or datetime.now(timezone.utc)
        record = self._store.get(user_id, doc_id)
        if record is None:
            raise ValueError(f"No record for user={user_id}, doc={doc_id}")

        if record.next_suggest_at == 0:
            # First decline: initial gap was threshold, next gap is threshold*2
            next_target = record.fetch_count + (self._threshold * 2)
        else:
            # Subsequent declines: double the gap from current fetch_count
            # to the target that was just declined
            prev_gap    = record.next_suggest_at - record.fetch_count
            next_target = record.fetch_count + max(prev_gap * 2, self._threshold * 2)

        record.user_state      = DocumentState.SURFACED
        record.suggestion_sent = False
        record.next_suggest_at = next_target
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
            record.fetch_count     = 0
            self._store.upsert(record)
        return record

    def _check_suggestion(self, record: UserDocRecord) -> Optional[SuggestionEvent]:
        """Return a SuggestionEvent if fetch_count has reached the current target."""
        if record.suggestion_sent:
            return None
        if record.user_state not in (DocumentState.SURFACED, DocumentState.SUGGESTED):
            return None

        target = record.next_suggest_at if record.next_suggest_at > 0 else self._threshold

        if record.fetch_count < target:
            return None

        # Mark as suggested
        record.user_state      = DocumentState.SUGGESTED
        record.suggestion_sent = True

        return SuggestionEvent(
            user_id     = record.user_id,
            doc_id      = record.doc_id,
            doc_title   = record.doc_title,
            doc_path    = record.doc_path,
            fetch_count = record.fetch_count,
        )
