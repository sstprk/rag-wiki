from storage.base import DocumentState, UserDocRecord
from datetime import datetime

TRANSITIONS = {
    DocumentState.GLOBAL:    [DocumentState.SURFACED],
    DocumentState.SURFACED:  [DocumentState.SUGGESTED, DocumentState.DEMOTED],
    DocumentState.SUGGESTED: [DocumentState.CLAIMED, DocumentState.SURFACED],
    DocumentState.CLAIMED:   [DocumentState.PINNED, DocumentState.DEMOTED],
    DocumentState.PINNED:    [DocumentState.CLAIMED],
    DocumentState.DEMOTED:   [DocumentState.SURFACED],
}

class StateMachine:
    def transition(self, record: UserDocRecord, 
                   to: DocumentState) -> UserDocRecord:
        allowed = TRANSITIONS.get(record.user_state, [])
        if to not in allowed:
            raise ValueError(
                f"Invalid transition: {record.user_state} → {to}"
            )
        record.user_state = to
        return record