"""
DecayEngine — computes relevance decay scores and applies PINNED/DEMOTED
transitions. Designed to run as a background scheduled job (daily).

Score formula:
  decay_score = weighted_avg(recency_factor, frequency_factor, explicit_signal)

  recency_factor  = exp(-λ * days_since_last_fetch)
  frequency_factor = min(fetch_count / FREQ_CAP, 1.0)
  explicit_signal  = value set by direct user actions (thumbs up/down etc.)
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from hybrid_kb.storage.base import DocumentState, StateStore, UserDocRecord
from hybrid_kb.lifecycle.state_machine import StateMachine


@dataclass
class DecayConfig:
    # Score weights (must sum to 1.0 conceptually — engine normalises)
    w_recency:   float = 0.5
    w_frequency: float = 0.3
    w_explicit:  float = 0.2

    # Exponential decay steepness (λ).  Half-life = ln(2)/λ ≈ 14 days at 0.05
    decay_lambda: float = 0.05

    # fetch_count is capped at this value when computing frequency factor
    freq_cap: int = 20

    # Thresholds for state transitions
    pin_threshold:       float = 0.85   # above this → PINNED
    demotion_threshold:  float = 0.15   # below this → DEMOTED

    # Hysteresis: decay score must hold for N days before pin/demote fires
    pin_hold_days:      int = 7
    demotion_hold_days: int = 3

    # Explicit signal bounds
    explicit_signal_min: float = 0.0
    explicit_signal_max: float = 1.0


@dataclass
class DecayResult:
    doc_id:     str
    old_state:  DocumentState
    new_state:  DocumentState
    old_score:  float
    new_score:  float
    transitioned: bool


class DecayEngine:

    def __init__(
        self,
        store:        StateStore,
        state_machine: StateMachine,
        config:       Optional[DecayConfig] = None,
    ):
        self._store = store
        self._sm    = state_machine
        self._cfg   = config or DecayConfig()

    # ─── Explicit signal API (called from user interaction events) ─────────────

    def thumbs_up(self, user_id: str, doc_id: str) -> None:
        """User explicitly marked this source as useful."""
        record = self._store.get(user_id, doc_id)
        if record is None:
            return
        record.explicit_signal = min(
            record.explicit_signal + 0.3,
            self._cfg.explicit_signal_max,
        )
        record.decay_score = self._compute_score(record)
        self._store.upsert(record)

    def thumbs_down(self, user_id: str, doc_id: str) -> None:
        """User explicitly marked this source as not useful."""
        record = self._store.get(user_id, doc_id)
        if record is None:
            return
        record.explicit_signal = max(
            record.explicit_signal - 0.3,
            self._cfg.explicit_signal_min,
        )
        record.decay_score = self._compute_score(record)
        self._store.upsert(record)

    def force_pin(self, user_id: str, doc_id: str) -> None:
        """User said 'always include this'. Suspend decay and pin."""
        record = self._store.get(user_id, doc_id)
        if record is None:
            return
        if record.user_state not in (DocumentState.CLAIMED, DocumentState.PINNED):
            return
        record = self._sm.transition(record, DocumentState.PINNED)
        record.explicit_signal = 1.0   # keep score high so decay doesn't fire
        self._store.upsert(record)

    def force_remove(self, user_id: str, doc_id: str) -> None:
        """User said 'remove from my KB'. Immediate demotion, block re-suggest 30d."""
        record = self._store.get(user_id, doc_id)
        if record is None:
            return
        now = datetime.utcnow()
        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)
        record.decay_score          = 0.0
        record.no_resiluggest_until = now + timedelta(days=30)
        self._store.upsert(record)

    # ─── Scheduled job entry point ─────────────────────────────────────────────

    def run_for_user(
        self,
        user_id: str,
        now: Optional[datetime] = None,
    ) -> list[DecayResult]:
        """
        Recompute decay scores for all CLAIMED + PINNED docs for this user.
        Apply state transitions where thresholds are crossed.

        Call once per day per user from your scheduler.
        """
        now     = now or datetime.utcnow()
        records = self._store.list_for_decay(user_id)
        results = []

        for record in records:
            old_state = record.user_state
            old_score = record.decay_score

            record.decay_score = self._compute_score(record, now)
            record = self._maybe_transition(record, now)

            self._store.upsert(record)
            results.append(DecayResult(
                doc_id      = record.doc_id,
                old_state   = old_state,
                new_state   = record.user_state,
                old_score   = old_score,
                new_score   = record.decay_score,
                transitioned = old_state != record.user_state,
            ))

        return results

    # ─── Score computation ─────────────────────────────────────────────────────

    def _compute_score(
        self,
        record: UserDocRecord,
        now: Optional[datetime] = None,
    ) -> float:
        now = now or datetime.utcnow()
        cfg = self._cfg

        # Recency factor — exponential decay from last fetch
        if record.last_fetched_at:
            days = (now - record.last_fetched_at).total_seconds() / 86400
            recency = math.exp(-cfg.decay_lambda * days)
        else:
            recency = 0.0

        # Frequency factor — normalised fetch count
        frequency = min(record.fetch_count / cfg.freq_cap, 1.0)

        # Explicit signal — already in [0, 1]
        explicit = max(
            cfg.explicit_signal_min,
            min(record.explicit_signal, cfg.explicit_signal_max),
        )

        # Weighted average
        total_weight = cfg.w_recency + cfg.w_frequency + cfg.w_explicit
        score = (
            cfg.w_recency   * recency  +
            cfg.w_frequency * frequency +
            cfg.w_explicit  * explicit
        ) / total_weight

        return round(max(0.0, min(1.0, score)), 4)

    # ─── State transitions ─────────────────────────────────────────────────────

    def _maybe_transition(
        self,
        record: UserDocRecord,
        now:    datetime,
    ) -> UserDocRecord:
        cfg   = self._cfg
        score = record.decay_score

        if record.user_state == DocumentState.CLAIMED:
            if score >= cfg.pin_threshold:
                if record.pinned_at is None:
                    record.pinned_at = now   # stamp candidate, transition next cycle
                else:
                    days_held = (now - record.pinned_at).days
                    if days_held >= cfg.pin_hold_days:
                        record = self._sm.transition(record, DocumentState.PINNED, now=now)
            elif score < cfg.demotion_threshold:
                if record.demoted_at is None:
                    record.demoted_at = now
                else:
                    days_held = (now - record.demoted_at).days
                    if days_held >= cfg.demotion_hold_days:
                        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)
            else:
                record.pinned_at  = None
                record.demoted_at = None

        elif record.user_state == DocumentState.PINNED:
            if score < cfg.pin_threshold:
                record.user_state = DocumentState.CLAIMED
                record.pinned_at  = None
            if score < cfg.demotion_threshold:
                if record.demoted_at is None:
                    record.demoted_at = now
                else:
                    days_held = (now - record.demoted_at).days
                    if days_held >= cfg.demotion_hold_days:
                        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)

        return record