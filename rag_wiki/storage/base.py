from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentState(str, Enum):
    GLOBAL    = "GLOBAL"
    SURFACED  = "SURFACED"
    SUGGESTED = "SUGGESTED"
    CLAIMED   = "CLAIMED"
    PINNED    = "PINNED"
    DEMOTED   = "DEMOTED"


@dataclass
class UserDocRecord:
    user_id:             str
    doc_id:              str
    doc_title:           str
    doc_path:            str
    user_state:          DocumentState = DocumentState.SURFACED
    fetch_count:         int           = 0
    last_fetched_at:     Optional[datetime] = None
    suggestion_sent:     bool          = False
    decay_score:         float         = 1.0
    explicit_signal:     float         = 0.0
    full_content:        Optional[str] = None   # populated on CLAIM
    cached_at:           Optional[datetime] = None
    pinned_at:           Optional[datetime] = None
    demoted_at:          Optional[datetime] = None
    no_resiluggest_until: Optional[datetime] = None  # after user declines
    next_suggest_at:     int           = 0    # fetch_count target for next suggestion (0 = use threshold)
    queries_missed:      int           = 0    # consecutive queries where doc was not retrieved


class StateStore(ABC):
    """
    Abstract contract for all state backends (SQLite, Postgres, etc).
    The rest of the system depends only on this interface.
    """

    @abstractmethod
    def get(self, user_id: str, doc_id: str) -> Optional[UserDocRecord]:
        """Return the record for this user+doc pair, or None if not tracked yet."""
        ...

    @abstractmethod
    def upsert(self, record: UserDocRecord) -> None:
        """Insert or update a record."""
        ...

    @abstractmethod
    def list_claimed(self, user_id: str) -> list[UserDocRecord]:
        """All CLAIMED docs for a user — used in cache-hit retrieval."""
        ...

    @abstractmethod
    def list_pinned(self, user_id: str) -> list[UserDocRecord]:
        """All PINNED docs for a user — injected into every context."""
        ...

    @abstractmethod
    def list_surfaced(self, user_id: str) -> list[UserDocRecord]:
        """All SURFACED and SUGGESTED docs for a user — used for miss tracking."""
        ...

    @abstractmethod
    def list_for_decay(self, user_id: str) -> list[UserDocRecord]:
        """CLAIMED + PINNED records eligible for decay scoring."""
        ...

    @abstractmethod
    def delete(self, user_id: str, doc_id: str) -> None:
        """Remove a record entirely (used on demotion to GLOBAL)."""
        ...

    @abstractmethod
    def list_active_users(self) -> list[str]:
        """Return distinct user_ids that have at least one CLAIMED or PINNED doc."""
        ...