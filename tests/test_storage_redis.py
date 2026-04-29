"""
Tests for RedisStateStore.
Uses fakeredis so no real Redis instance is needed.
"""

import pytest
from datetime import datetime, timezone

import fakeredis

from rag_wiki.storage.base import DocumentState, UserDocRecord
from rag_wiki.storage.redis_store import RedisStateStore


@pytest.fixture
def store() -> RedisStateStore:
    """Fresh RedisStateStore backed by a fakeredis instance."""
    client = fakeredis.FakeRedis()
    return RedisStateStore(client)


def make_record(**overrides) -> UserDocRecord:
    """Create a UserDocRecord with sensible defaults."""
    defaults = dict(
        user_id="user-1",
        doc_id="doc-1",
        doc_title="Test Doc",
        doc_path="/kb/test.pdf",
        user_state=DocumentState.SURFACED,
        fetch_count=0,
    )
    defaults.update(overrides)
    return UserDocRecord(**defaults)


# ─── Get / Upsert ─────────────────────────────────────────────────────────────


class TestGetUpsert:
    def test_get_missing_returns_none(self, store: RedisStateStore) -> None:
        assert store.get("user-1", "doc-1") is None

    def test_upsert_then_get(self, store: RedisStateStore) -> None:
        r = make_record()
        store.upsert(r)
        fetched = store.get("user-1", "doc-1")
        assert fetched is not None
        assert fetched.doc_title == "Test Doc"
        assert fetched.user_state == DocumentState.SURFACED

    def test_upsert_updates_existing(self, store: RedisStateStore) -> None:
        r = make_record()
        store.upsert(r)
        r.fetch_count = 5
        r.user_state = DocumentState.CLAIMED
        store.upsert(r)
        fetched = store.get("user-1", "doc-1")
        assert fetched.fetch_count == 5
        assert fetched.user_state == DocumentState.CLAIMED

    def test_different_users_isolated(self, store: RedisStateStore) -> None:
        r1 = make_record(user_id="user-1", doc_id="doc-1")
        r2 = make_record(user_id="user-2", doc_id="doc-1", doc_title="Other")
        store.upsert(r1)
        store.upsert(r2)
        assert store.get("user-1", "doc-1").doc_title == "Test Doc"
        assert store.get("user-2", "doc-1").doc_title == "Other"

    def test_roundtrip_preserves_datetime_fields(self, store: RedisStateStore) -> None:
        now = datetime.now(timezone.utc)
        r = make_record(
            last_fetched_at=now,
            cached_at=now,
            pinned_at=now,
        )
        store.upsert(r)
        fetched = store.get("user-1", "doc-1")
        assert fetched.last_fetched_at is not None
        assert fetched.cached_at is not None
        assert fetched.pinned_at is not None

    def test_roundtrip_preserves_full_content(self, store: RedisStateStore) -> None:
        r = make_record(full_content="Lorem ipsum dolor sit amet")
        store.upsert(r)
        fetched = store.get("user-1", "doc-1")
        assert fetched.full_content == "Lorem ipsum dolor sit amet"


# ─── List methods ─────────────────────────────────────────────────────────────


class TestListMethods:
    def test_list_claimed_empty(self, store: RedisStateStore) -> None:
        assert store.list_claimed("user-1") == []

    def test_list_claimed_returns_only_claimed(self, store: RedisStateStore) -> None:
        store.upsert(make_record(doc_id="doc-1", user_state=DocumentState.CLAIMED))
        store.upsert(make_record(doc_id="doc-2", user_state=DocumentState.SURFACED))
        store.upsert(make_record(doc_id="doc-3", user_state=DocumentState.PINNED))
        claimed = store.list_claimed("user-1")
        assert len(claimed) == 1
        assert claimed[0].doc_id == "doc-1"

    def test_list_pinned_returns_only_pinned(self, store: RedisStateStore) -> None:
        store.upsert(make_record(doc_id="doc-1", user_state=DocumentState.PINNED))
        store.upsert(make_record(doc_id="doc-2", user_state=DocumentState.CLAIMED))
        pinned = store.list_pinned("user-1")
        assert len(pinned) == 1
        assert pinned[0].doc_id == "doc-1"

    def test_list_for_decay_returns_claimed_and_pinned(
        self, store: RedisStateStore
    ) -> None:
        store.upsert(make_record(doc_id="doc-1", user_state=DocumentState.CLAIMED))
        store.upsert(make_record(doc_id="doc-2", user_state=DocumentState.PINNED))
        store.upsert(make_record(doc_id="doc-3", user_state=DocumentState.SURFACED))
        decay_list = store.list_for_decay("user-1")
        ids = {r.doc_id for r in decay_list}
        assert ids == {"doc-1", "doc-2"}

    def test_lists_are_user_scoped(self, store: RedisStateStore) -> None:
        store.upsert(make_record(user_id="user-1", doc_id="doc-1", user_state=DocumentState.CLAIMED))
        store.upsert(make_record(user_id="user-2", doc_id="doc-1", user_state=DocumentState.CLAIMED))
        assert len(store.list_claimed("user-1")) == 1
        assert len(store.list_claimed("user-2")) == 1

    def test_state_index_updated_on_state_change(self, store: RedisStateStore) -> None:
        """When a record's state changes, it should move between index sets."""
        store.upsert(make_record(doc_id="doc-1", user_state=DocumentState.CLAIMED))
        assert len(store.list_claimed("user-1")) == 1

        r = store.get("user-1", "doc-1")
        r.user_state = DocumentState.PINNED
        store.upsert(r)
        assert len(store.list_claimed("user-1")) == 0
        assert len(store.list_pinned("user-1")) == 1


# ─── Delete ────────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_removes_record(self, store: RedisStateStore) -> None:
        store.upsert(make_record())
        store.delete("user-1", "doc-1")
        assert store.get("user-1", "doc-1") is None

    def test_delete_nonexistent_is_noop(self, store: RedisStateStore) -> None:
        store.delete("user-1", "doc-999")  # should not raise

    def test_delete_cleans_state_index(self, store: RedisStateStore) -> None:
        store.upsert(make_record(doc_id="doc-1", user_state=DocumentState.CLAIMED))
        store.delete("user-1", "doc-1")
        assert len(store.list_claimed("user-1")) == 0


# ─── list_active_users ─────────────────────────────────────────────────────────


class TestListActiveUsers:
    def test_empty_store_returns_empty(self, store: RedisStateStore) -> None:
        assert store.list_active_users() == []

    def test_returns_users_with_claimed_or_pinned(
        self, store: RedisStateStore
    ) -> None:
        store.upsert(make_record(user_id="alice", doc_id="d1", user_state=DocumentState.CLAIMED))
        store.upsert(make_record(user_id="bob", doc_id="d2", user_state=DocumentState.PINNED))
        store.upsert(make_record(user_id="carol", doc_id="d3", user_state=DocumentState.SURFACED))
        users = store.list_active_users()
        assert "alice" in users
        assert "bob" in users
        assert "carol" not in users

    def test_returns_sorted_unique_users(self, store: RedisStateStore) -> None:
        store.upsert(make_record(user_id="zara", doc_id="d1", user_state=DocumentState.CLAIMED))
        store.upsert(make_record(user_id="zara", doc_id="d2", user_state=DocumentState.PINNED))
        store.upsert(make_record(user_id="alice", doc_id="d3", user_state=DocumentState.CLAIMED))
        users = store.list_active_users()
        assert users == ["alice", "zara"]
