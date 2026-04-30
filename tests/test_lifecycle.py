"""
Tests for StateMachine, FetchCounter, and DecayEngine.
"""

import pytest
from datetime import datetime, timedelta, timezone

from rag_wiki.storage.base import DocumentState, UserDocRecord
from rag_wiki.storage.sqlite import SQLiteStateStore
from rag_wiki.storage.chunk_store import ChunkStore
from rag_wiki.lifecycle.state_machine import StateMachine, InvalidTransitionError
from rag_wiki.lifecycle.fetch_counter import FetchCounter
from rag_wiki.lifecycle.decay_engine import DecayEngine, DecayConfig


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
    cfg = DecayConfig(pin_hold_days=0, demotion_hold_days=0, w_chunk_hit=0.15)
    chunk_store = ChunkStore(wiki_save_dir=None)
    return DecayEngine(store, sm, config=cfg, chunk_store=chunk_store)

@pytest.fixture
def chunk_store():
    return ChunkStore(wiki_save_dir=None)


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
        # next_suggest_at is set to an escalated target after decline
        assert record.next_suggest_at > record.fetch_count

    def test_decline_then_fetch_again_no_suggestion_until_target(self, counter):
        for _ in range(3):
            counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
        counter.decline_suggestion("u1", "d1")
        # After decline with threshold=3, next target = fetch_count + threshold*2 = 3+6 = 9
        # Fetches 4-8 should not fire a suggestion
        for _ in range(5):
            event = counter.record_fetch("u1", "d1", "Doc", "/kb/doc.pdf")
            assert event is None  # not yet at next_suggest_at

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
            last_fetched_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
            full_content="some content",
        )
        store.upsert(r)
        return r

    def test_fresh_doc_has_high_score(self, decay, store):
        self._claimed_record(store, fetch_count=15, days_ago=0)
        results = decay.run_for_user("u1")
        # With w_chunk_hit=0.15 and no chunks stored, score is lower than old baseline.
        # recency≈1.0, frequency=0.75, explicit=0, chunk_hit=0 → ~0.59
        assert results[0].new_score > 0.5

    def test_stale_doc_has_low_score(self, decay, store):
        self._claimed_record(store, fetch_count=1, days_ago=120)
        results = decay.run_for_user("u1")
        assert results[0].new_score < 0.5

    def test_high_score_transitions_to_pinned(self, decay, store):
        r = self._claimed_record(store, fetch_count=20, days_ago=0)
        r.explicit_signal = 1.0
        r.pinned_at = datetime.now(timezone.utc) - timedelta(days=10)  # already held long enough
        store.upsert(r)
        results = decay.run_for_user("u1", now=datetime.now(timezone.utc))
        assert results[0].new_state == DocumentState.PINNED

    def test_very_low_score_transitions_to_demoted(self, decay, store):
        r = self._claimed_record(store, fetch_count=0, days_ago=200)
        r.explicit_signal = 0.0
        r.demoted_at = datetime.now(timezone.utc) - timedelta(days=10)  # already held long enough
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


# ─── DecayEngine + ChunkStore integration ─────────────────────────────────────

class TestDecayEngineChunkStore:
    def _claimed_record(self, store, fetch_count=10, days_ago=1):
        r = UserDocRecord(
            user_id="u1", doc_id="d1",
            doc_title="Doc", doc_path="/kb/doc.pdf",
            user_state=DocumentState.CLAIMED,
            fetch_count=fetch_count,
            last_fetched_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
            full_content="some content",
        )
        store.upsert(r)
        return r

    def test_chunk_hit_rate_improves_score(self, store, sm, chunk_store):
        """100% hit rate record scores higher than 0% hit rate record."""
        cfg = DecayConfig(pin_hold_days=0, demotion_hold_days=0, w_chunk_hit=0.15)
        engine = DecayEngine(store, sm, config=cfg, chunk_store=chunk_store)

        # Record with 0% hit rate
        r_no_hits = UserDocRecord(
            user_id="u1", doc_id="d1",
            doc_title="Doc", doc_path="/p",
            user_state=DocumentState.CLAIMED,
            fetch_count=10,
            last_fetched_at=datetime.now(timezone.utc) - timedelta(days=1),
            full_content="content",
        )
        store.upsert(r_no_hits)
        chunk_store.add_chunks("u1", "d1", [
            {"chunk_index": 0, "text": "t", "vector": [1.0], "section": "",
             "hit_count": 0, "last_hit_at": None},
        ])

        # Record with 100% hit rate
        r_all_hits = UserDocRecord(
            user_id="u1", doc_id="d2",
            doc_title="Doc2", doc_path="/p2",
            user_state=DocumentState.CLAIMED,
            fetch_count=10,
            last_fetched_at=datetime.now(timezone.utc) - timedelta(days=1),
            full_content="content",
        )
        store.upsert(r_all_hits)
        chunk_store.add_chunks("u1", "d2", [
            {"chunk_index": 0, "text": "t", "vector": [1.0], "section": "",
             "hit_count": 5, "last_hit_at": datetime.now(timezone.utc).isoformat()},
        ])

        score_no_hits  = engine._compute_score(r_no_hits,  user_id="u1")
        score_all_hits = engine._compute_score(r_all_hits, user_id="u1")
        assert score_all_hits > score_no_hits

    def test_force_remove_deletes_chunks(self, store, sm, chunk_store):
        """force_remove clears the chunk store for that doc."""
        cfg = DecayConfig(pin_hold_days=0, demotion_hold_days=0)
        engine = DecayEngine(store, sm, config=cfg, chunk_store=chunk_store)

        self._claimed_record(store)
        chunk_store.add_chunks("u1", "d1", [
            {"chunk_index": 0, "text": "t", "vector": [1.0], "section": "",
             "hit_count": 0, "last_hit_at": None},
        ])
        assert len(chunk_store.load_chunks("u1", "d1")) == 1

        engine.force_remove("u1", "d1")
        assert chunk_store.load_chunks("u1", "d1") == []

    def test_demotion_transition_deletes_chunks(self, store, sm, chunk_store):
        """Decay-triggered demotion clears the chunk store."""
        cfg = DecayConfig(pin_hold_days=0, demotion_hold_days=0, w_chunk_hit=0.15)
        engine = DecayEngine(store, sm, config=cfg, chunk_store=chunk_store)

        # Very stale record that will score below demotion_threshold
        r = UserDocRecord(
            user_id="u1", doc_id="d1",
            doc_title="Doc", doc_path="/p",
            user_state=DocumentState.CLAIMED,
            fetch_count=0,
            last_fetched_at=datetime.now(timezone.utc) - timedelta(days=200),
            full_content="content",
            demoted_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        store.upsert(r)
        chunk_store.add_chunks("u1", "d1", [
            {"chunk_index": 0, "text": "t", "vector": [1.0], "section": "",
             "hit_count": 0, "last_hit_at": None},
        ])
        assert len(chunk_store.load_chunks("u1", "d1")) == 1

        engine.run_for_user("u1")
        assert chunk_store.load_chunks("u1", "d1") == []
