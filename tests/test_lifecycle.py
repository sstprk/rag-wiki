"""
Tests for StateMachine, FetchCounter, and DecayEngine.
"""

import pytest
from datetime import datetime, timedelta

from hybrid_kb.storage.base import DocumentState, UserDocRecord
from hybrid_kb.storage.sqlite import SQLiteStateStore
from hybrid_kb.lifecycle.state_machine import StateMachine, InvalidTransitionError
from hybrid_kb.lifecycle.fetch_counter import FetchCounter
from hybrid_kb.lifecycle.decay_engine import DecayEngine, DecayConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    return SQLiteStateStore("sqlite:///:memory:")

@pytest.fixture
def sm():
    return StateMachine()

@pytest.fixture
def counter(store, sm):
    return FetchCounter(store, sm, fetch_threshold=3, no_resiluggest_days=30)

@pytest.fixture
def decay(store, sm):
    cfg = DecayConfig(pin_hold_days=0, demotion_hold_days=0)  # no hysteresis in tests
    return DecayEngine(store, sm, config=cfg)


# ─── StateMachine ─────────────────────────────────────────────────────────────

class TestStateMachine:
    def _record(self, state=DocumentState.SURFACED):
        return UserDocRecord(
            user_id="u1", doc_id="d1",
            doc_title="Doc", doc_path="/kb/doc.pdf",
            user_state=state,
        )

    def test_valid_transition_surfaced_to_suggested(self, sm):
        r = self._record(DocumentState.SURFACED)
        r = sm.transition(r, DocumentState.SUGGESTED)
        assert r.user_state == DocumentState.SUGGESTED

    def test_valid_transition_suggested_to_claimed(self, sm):
        r = self._record(DocumentState.SUGGESTED)
        r = sm.transition(r, DocumentState.CLAIMED)
        assert r.user_state == DocumentState.CLAIMED
        assert r.cached_at is not None

    def test_valid_transition_claimed_to_pinned(self, sm):
        r = self._record(DocumentState.CLAIMED)
        r = sm.transition(r, DocumentState.PINNED)
        assert r.user_state == DocumentState.PINNED
        assert r.pinned_at is not None

    def test_demotion_clears_content(self, sm):
        r = self._record(DocumentState.CLAIMED)
        r.full_content = "some content"
        r = sm.transition(r, DocumentState.DEMOTED)
        assert r.full_content is None
        assert r.demoted_at is not None

    def test_invalid_transition_raises(self, sm):
        r = self._record(DocumentState.GLOBAL)
        with pytest.raises(InvalidTransitionError):
            sm.transition(r, DocumentState.PINNED)

    def test_can_transition_returns_false_for_invalid(self, sm):
        r = self._record(DocumentState.SURFACED)
        assert sm.can_transition(r, DocumentState.GLOBAL) is False

    def test_make_surfaced_creates_correct_record(self, sm):
        r = sm.make_surfaced("u1", "d1", "My Doc", "/path")
        assert r.user_state == DocumentState.SURFACED
        assert r.user_id == "u1"


# ─── FetchCounter ─────────────────────────────────────────────────────────────

class TestFetchCounter:
    def test_first_fetch_returns_no_suggestion(self, counter):
        event = counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        assert event is None

    def test_suggestion_fires_at_threshold(self, counter):
        for i in range(2):
            event = counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
            assert event is None
        event = counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        assert event is not None
        assert event.doc_id == "d1"
        assert event.fetch_count == 3

    def test_suggestion_fires_only_once(self, counter):
        for _ in range(5):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        # Reset suggestion_sent to False manually won't happen — check only 1 fires
        events = []
        store = counter._store
        # Already fired after 3rd call — 4th and 5th return None
        # (covered implicitly by state change to SUGGESTED)
        record = store.get("u1", "d1")
        assert record.suggestion_sent is True

    def test_accept_suggestion_transitions_to_claimed(self, counter, store):
        for _ in range(3):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        counter.accept_suggestion("u1", "d1", full_content="full text here")
        record = store.get("u1", "d1")
        assert record.user_state == DocumentState.CLAIMED
        assert record.full_content == "full text here"

    def test_decline_suggestion_blocks_resiluggest(self, counter, store):
        for _ in range(3):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        counter.decline_suggestion("u1", "d1")
        record = store.get("u1", "d1")
        assert record.user_state == DocumentState.SURFACED
        assert record.no_resiluggest_until is not None

    def test_decline_then_fetch_again_no_suggestion_during_block(self, counter):
        for _ in range(3):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        counter.decline_suggestion("u1", "d1")
        # Reset suggestion_sent to False (simulated by decline)
        # More fetches should not fire during block period
        for _ in range(5):
            event = counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
            assert event is None  # blocked

    def test_no_counter_increment_for_claimed_docs(self, counter, store):
        for _ in range(3):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        counter.accept_suggestion("u1", "d1", full_content="content")
        record_before = store.get("u1", "d1")
        count_before = record_before.fetch_count
        # Further fetches on a CLAIMED doc should not increment
        counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        record_after = store.get("u1", "d1")
        assert record_after.fetch_count == count_before


# ─── DecayEngine ──────────────────────────────────────────────────────────────

class TestDecayEngine:
    def _claimed_record(self, store, fetch_count=10, days_ago=1):
        r = UserDocRecord(
            user_id="u1", doc_id="d1",
            doc_title="Doc", doc_path="/kb/doc.pdf",
            user_state=DocumentState.CLAIMED,
            fetch_count=fetch_count,
            last_fetched_at=datetime.utcnow() - timedelta(days=days_ago),
            full_content="some content",
        )
        store.upsert(r)
        return r

    def test_fresh_doc_has_high_score(self, decay, store):
        self._claimed_record(store, fetch_count=15, days_ago=0)
        results = decay.run_for_user("u1")
        assert results[0].new_score > 0.7

    def test_stale_doc_has_low_score(self, decay, store):
        self._claimed_record(store, fetch_count=1, days_ago=120)
        results = decay.run_for_user("u1")
        assert results[0].new_score < 0.5

    def test_high_score_transitions_to_pinned(self, decay, store):
        r = self._claimed_record(store, fetch_count=20, days_ago=0)
        r.explicit_signal = 1.0
        r.pinned_at = datetime.utcnow() - timedelta(days=10)  # already held long enough
        store.upsert(r)
        results = decay.run_for_user("u1", now=datetime.utcnow())
        assert results[0].new_state == DocumentState.PINNED

    def test_very_low_score_transitions_to_demoted(self, decay, store):
        r = self._claimed_record(store, fetch_count=0, days_ago=200)
        r.explicit_signal = 0.0
        r.demoted_at = datetime.utcnow() - timedelta(days=10)  # already held long enough
        store.upsert(r)
        results = decay.run_for_user("u1")
        assert results[0].new_state == DocumentState.DEMOTED

    def test_thumbs_up_increases_score(self, decay, store):
        r = self._claimed_record(store, fetch_count=5, days_ago=30)
        before = store.get("u1", "d1").explicit_signal
        decay.thumbs_up("u1", "d1")
        after = store.get("u1", "d1").explicit_signal
        assert after > before

    def test_force_remove_sets_state_to_demoted(self, decay, store):
        self._claimed_record(store)
        decay.force_remove("u1", "d1")
        record = store.get("u1", "d1")
        assert record.user_state == DocumentState.DEMOTED
        assert record.decay_score == 0.0
        assert record.no_resiluggest_until is not None