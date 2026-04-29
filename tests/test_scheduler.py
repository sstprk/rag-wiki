"""
Tests for DecayScheduler.
Uses mocked DecayEngine and MemoryStateStore — no real scheduling waits.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, call

from rag_wiki.storage.base import DocumentState, UserDocRecord
from rag_wiki.storage.memory import MemoryStateStore
from rag_wiki.lifecycle.decay_engine import DecayEngine
from rag_wiki.scheduler import DecayScheduler


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_record(**overrides) -> UserDocRecord:
    """Create a UserDocRecord with sensible defaults."""
    defaults = dict(
        user_id="user-1",
        doc_id="doc-1",
        doc_title="Test Doc",
        doc_path="/kb/test.pdf",
        user_state=DocumentState.CLAIMED,
        fetch_count=5,
    )
    defaults.update(overrides)
    return UserDocRecord(**defaults)


@pytest.fixture
def store() -> MemoryStateStore:
    return MemoryStateStore()


@pytest.fixture
def mock_engine() -> MagicMock:
    """A mock DecayEngine — records calls but does nothing."""
    engine = MagicMock(spec=DecayEngine)
    engine.run_for_user.return_value = []
    return engine


# ─── run_now ───────────────────────────────────────────────────────────────────


class TestRunNow:
    def test_run_now_calls_engine(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        scheduler = DecayScheduler(mock_engine, store, backend="simple")
        scheduler.run_now("user-42")
        mock_engine.run_for_user.assert_called_once_with("user-42")


# ─── run_all_users ─────────────────────────────────────────────────────────────


class TestRunAllUsers:
    def test_run_all_users_iterates_active_users(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        store.upsert(make_record(user_id="alice", doc_id="d1"))
        store.upsert(make_record(user_id="bob", doc_id="d2", user_state=DocumentState.PINNED))
        store.upsert(make_record(user_id="carol", doc_id="d3", user_state=DocumentState.SURFACED))

        scheduler = DecayScheduler(mock_engine, store, backend="simple")
        scheduler.run_all_users()

        called_users = {c.args[0] for c in mock_engine.run_for_user.call_args_list}
        assert "alice" in called_users
        assert "bob" in called_users
        assert "carol" not in called_users

    def test_run_all_users_empty_store(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        scheduler = DecayScheduler(mock_engine, store, backend="simple")
        scheduler.run_all_users()
        mock_engine.run_for_user.assert_not_called()


# ─── Simple backend lifecycle ─────────────────────────────────────────────────


class TestSimpleBackend:
    def test_start_stop_lifecycle(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        scheduler = DecayScheduler(
            mock_engine, store, backend="simple", interval_hours=1
        )
        assert not scheduler.is_running

        scheduler.start()
        assert scheduler.is_running
        assert scheduler._timer is not None

        scheduler.stop()
        assert not scheduler.is_running

    def test_double_start_is_idempotent(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        scheduler = DecayScheduler(mock_engine, store, backend="simple")
        scheduler.start()
        timer1 = scheduler._timer
        scheduler.start()  # should not create a second timer
        assert scheduler._timer is timer1
        scheduler.stop()

    def test_simple_tick_calls_run_all_users(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        """Directly invoke the tick method to verify it calls run_all_users."""
        store.upsert(make_record(user_id="u1", doc_id="d1"))
        scheduler = DecayScheduler(
            mock_engine, store, backend="simple", interval_hours=24
        )
        scheduler._running = True
        # Call tick directly (don't wait for timer)
        scheduler._simple_tick()
        mock_engine.run_for_user.assert_called_once_with("u1")
        scheduler.stop()

    def test_tick_reschedules(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        """After a tick, a new timer should be scheduled."""
        scheduler = DecayScheduler(
            mock_engine, store, backend="simple", interval_hours=24
        )
        scheduler._running = True
        scheduler._simple_tick()
        assert scheduler._timer is not None
        scheduler.stop()


# ─── APScheduler backend ──────────────────────────────────────────────────────


class TestAPSchedulerBackend:
    def test_apscheduler_start_stop(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        """Test that APScheduler backend starts and stops cleanly."""
        try:
            import apscheduler  # noqa: F401
        except ImportError:
            pytest.skip("apscheduler not installed")

        scheduler = DecayScheduler(
            mock_engine, store, backend="apscheduler", interval_hours=1
        )
        scheduler.start()
        assert scheduler.is_running
        assert scheduler._aps_scheduler is not None

        scheduler.stop()
        assert not scheduler.is_running

    def test_apscheduler_import_error(
        self, mock_engine: MagicMock, store: MemoryStateStore
    ) -> None:
        """If apscheduler is not installed, start() should raise ImportError."""
        scheduler = DecayScheduler(
            mock_engine, store, backend="apscheduler"
        )
        with patch.dict("sys.modules", {"apscheduler": None, "apscheduler.schedulers": None,
                                         "apscheduler.schedulers.background": None,
                                         "apscheduler.triggers": None,
                                         "apscheduler.triggers.interval": None}):
            with pytest.raises(ImportError, match="apscheduler"):
                scheduler.start()
