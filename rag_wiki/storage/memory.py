"""
MemoryStateStore — zero-dependency, in-memory backend.

Backed by a plain Python dict keyed by ``(user_id, doc_id)`` tuples.
All public methods are protected by a :class:`threading.Lock` for
thread-safety in multi-threaded environments (e.g. web servers).

Perfect for tests, prototyping, and single-process deployments where
persistence across restarts is not required.
"""

from __future__ import annotations

import copy
import threading
from typing import Optional

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord


class MemoryStateStore(StateStore):
    """
    In-memory implementation of :class:`StateStore`.

    All data lives in a plain ``dict`` and is lost when the process exits.
    All operations are protected by a :class:`threading.Lock`.

    Usage::

        store = MemoryStateStore()
        store.upsert(record)
        store.get("user-1", "doc-1")
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], UserDocRecord] = {}
        self._lock = threading.Lock()

    # ─── StateStore interface ──────────────────────────────────────────────────

    def get(self, user_id: str, doc_id: str) -> Optional[UserDocRecord]:
        """Return the record for this user+doc pair, or ``None`` if not tracked."""
        with self._lock:
            record = self._data.get((user_id, doc_id))
            return copy.deepcopy(record) if record is not None else None

    def upsert(self, record: UserDocRecord) -> None:
        """Insert or update a record (full replacement on the key)."""
        with self._lock:
            self._data[(record.user_id, record.doc_id)] = copy.deepcopy(record)

    def list_claimed(self, user_id: str) -> list[UserDocRecord]:
        """All CLAIMED docs for a user — used in cache-hit retrieval."""
        with self._lock:
            return [
                copy.deepcopy(r)
                for r in self._data.values()
                if r.user_id == user_id and r.user_state == DocumentState.CLAIMED
            ]

    def list_pinned(self, user_id: str) -> list[UserDocRecord]:
        """All PINNED docs for a user — injected into every context."""
        with self._lock:
            return [
                copy.deepcopy(r)
                for r in self._data.values()
                if r.user_id == user_id and r.user_state == DocumentState.PINNED
            ]

    def list_surfaced(self, user_id: str) -> list[UserDocRecord]:
        """All SURFACED and SUGGESTED docs — used for miss tracking."""
        with self._lock:
            return [
                copy.deepcopy(r)
                for r in self._data.values()
                if r.user_id == user_id
                and r.user_state in (DocumentState.SURFACED, DocumentState.SUGGESTED)
            ]

    def list_for_decay(self, user_id: str) -> list[UserDocRecord]:
        """CLAIMED + PINNED records eligible for decay scoring."""
        with self._lock:
            return [
                copy.deepcopy(r)
                for r in self._data.values()
                if r.user_id == user_id
                and r.user_state in (DocumentState.CLAIMED, DocumentState.PINNED)
            ]

    def delete(self, user_id: str, doc_id: str) -> None:
        """Remove a record entirely (used on demotion to GLOBAL)."""
        with self._lock:
            self._data.pop((user_id, doc_id), None)

    def list_active_users(self) -> list[str]:
        """Return distinct user_ids that have at least one CLAIMED or PINNED doc."""
        with self._lock:
            users: set[str] = set()
            for (uid, _), record in self._data.items():
                if record.user_state in (DocumentState.CLAIMED, DocumentState.PINNED):
                    users.add(uid)
            return sorted(users)
