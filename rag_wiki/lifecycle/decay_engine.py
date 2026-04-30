"""
DecayEngine — computes relevance decay scores and applies PINNED/DEMOTED
transitions. Designed to run as a background scheduled job (daily).

Score formula:
  decay_score = weighted_avg(recency, frequency, explicit_signal, chunk_hit_rate)

  recency_factor   = exp(-λ * days_since_last_fetch)
  frequency_factor = min(fetch_count / FREQ_CAP, 1.0)
  explicit_signal  = value set by direct user actions (thumbs up/down etc.)
  chunk_hit_rate   = fraction of cached chunks ever matched by a query
"""

import math
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
class DecayConfig:
    # Score weights (engine normalises by total_weight so exact sum is flexible)
    w_recency:    float = 0.40
    w_frequency:  float = 0.30
    w_explicit:   float = 0.20
    w_chunk_hit:  float = 0.15

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

    # Immediate demotion after this many consecutive cache misses
    max_cache_miss_streak: int = 10


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
        store:         StateStore,
        state_machine: StateMachine,
        config:        Optional[DecayConfig] = None,
        chunk_store:   Optional[object] = None,
    ):
        self._store       = store
        self._sm          = state_machine
        self._cfg         = config or DecayConfig()
        self._chunk_store = chunk_store

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
        record.decay_score = self._compute_score(record, user_id=user_id)
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
        record.decay_score = self._compute_score(record, user_id=user_id)
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
        now = datetime.now(timezone.utc)
        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)
        record.decay_score          = 0.0
        record.no_resiluggest_until = now + timedelta(days=30)
        self._store.upsert(record)
        if self._chunk_store is not None:
            self._chunk_store.delete(user_id, doc_id)

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
        now     = now or datetime.now(timezone.utc)
        records = self._store.list_for_decay(user_id)
        results = []

        for record in records:
            old_state = record.user_state
            old_score = record.decay_score

            record.decay_score = self._compute_score(record, now, user_id=user_id)
            record = self._maybe_transition(record, now)

            self._store.upsert(record)
            results.append(DecayResult(
                doc_id       = record.doc_id,
                old_state    = old_state,
                new_state    = record.user_state,
                old_score    = old_score,
                new_score    = record.decay_score,
                transitioned = old_state != record.user_state,
            ))

        return results

    # ─── Score computation ─────────────────────────────────────────────────────

    def _compute_score(
        self,
        record:  UserDocRecord,
        now:     Optional[datetime] = None,
        user_id: Optional[str]      = None,
    ) -> float:
        now = _ensure_utc(now) or datetime.now(timezone.utc)
        cfg = self._cfg

        # Recency factor — exponential decay from last fetch
        last_fetched = _ensure_utc(record.last_fetched_at)
        if last_fetched:
            days = (now - last_fetched).total_seconds() / 86400
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

        # Chunk hit rate — fraction of cached chunks ever matched by a query
        if self._chunk_store is not None and user_id is not None:
            chunk_hit_rate = self._chunk_store.get_hit_rate(user_id, record.doc_id)
        else:
            chunk_hit_rate = 0.0

        # Weighted average (engine normalises so weights don't need to sum to 1)
        total_weight = cfg.w_recency + cfg.w_frequency + cfg.w_explicit + cfg.w_chunk_hit
        score = (
            cfg.w_recency    * recency         +
            cfg.w_frequency  * frequency       +
            cfg.w_explicit   * explicit        +
            cfg.w_chunk_hit  * chunk_hit_rate
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
                    days_held = (now - _ensure_utc(record.pinned_at)).days
                    if days_held >= cfg.pin_hold_days:
                        record = self._sm.transition(record, DocumentState.PINNED, now=now)
                        if self._chunk_store is not None:
                            self._chunk_store.delete(record.user_id, record.doc_id)
            elif score < cfg.demotion_threshold:
                if record.demoted_at is None:
                    record.demoted_at = now
                else:
                    days_held = (now - _ensure_utc(record.demoted_at)).days
                    if days_held >= cfg.demotion_hold_days:
                        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)
                        if self._chunk_store is not None:
                            self._chunk_store.delete(record.user_id, record.doc_id)
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
                    days_held = (now - _ensure_utc(record.demoted_at)).days
                    if days_held >= cfg.demotion_hold_days:
                        record = self._sm.transition(record, DocumentState.DEMOTED, now=now)
                        if self._chunk_store is not None:
                            self._chunk_store.delete(record.user_id, record.doc_id)

        return record
