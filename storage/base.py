from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    user_id:          str
    doc_id:           str
    doc_title:        str
    doc_path:         str
    user_state:       DocumentState
    fetch_count:      int
    last_fetched_at:  Optional[datetime]
    suggestion_sent:  bool
    decay_score:      float
    explicit_signal:  float
    full_content:     Optional[str]   # populated when CLAIMED

class StateStore(ABC):
    @abstractmethod
    def get(self, user_id: str, doc_id: str) -> Optional[UserDocRecord]: ...

    @abstractmethod
    def upsert(self, record: UserDocRecord) -> None: ...

    @abstractmethod
    def list_claimed(self, user_id: str) -> list[UserDocRecord]: ...

    @abstractmethod
    def list_pinned(self, user_id: str) -> list[UserDocRecord]: ...

    @abstractmethod
    def list_for_decay(self, user_id: str) -> list[UserDocRecord]: ...