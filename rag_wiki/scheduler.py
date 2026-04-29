"""
DecayScheduler — runs :meth:`DecayEngine.run_for_user` on a configurable
interval. Supports two backends:

- ``"simple"`` — a plain :class:`threading.Timer` loop (zero extra deps)
- ``"apscheduler"`` — uses APScheduler :class:`BackgroundScheduler`

Usage::

    scheduler = DecayScheduler(decay_engine, store, backend="simple",
                               interval_hours=24)
    scheduler.start()
    scheduler.run_now("user-123")   # manual trigger
    scheduler.run_all_users()       # iterate over all active users
    scheduler.stop()
"""

from __future__ import annotations

import logging
import threading
from typing import Literal, Optional

from rag_wiki.lifecycle.decay_engine import DecayEngine
from rag_wiki.storage.base import StateStore

logger = logging.getLogger(__name__)


class DecayScheduler:
    """
    Wraps a :class:`DecayEngine` and calls :meth:`run_for_user` on a
    schedule.

    Parameters
    ----------
    decay_engine:
        The engine that recomputes decay scores.
    store:
        StateStore used to discover active user ids.
    backend:
        ``"simple"`` for a plain :class:`threading.Timer` loop, or
        ``"apscheduler"`` for APScheduler's BackgroundScheduler.
    interval_hours:
        How often the decay job runs (default 24 h).
    """

    def __init__(
        self,
        decay_engine: DecayEngine,
        store: StateStore,
        *,
        backend: Literal["simple", "apscheduler"] = "simple",
        interval_hours: float = 24,
    ) -> None:
        self._engine = decay_engine
        self._store = store
        self._backend = backend
        self._interval_seconds = interval_hours * 3600
        self._running = False
        self._timer: Optional[threading.Timer] = None
        self._aps_scheduler: Optional[object] = None  # BackgroundScheduler

    # ─── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background decay scheduler."""
        if self._running:
            return
        self._running = True

        if self._backend == "apscheduler":
            self._start_apscheduler()
        else:
            self._start_simple()

    def stop(self) -> None:
        """Stop the background decay scheduler."""
        self._running = False

        if self._backend == "apscheduler" and self._aps_scheduler is not None:
            self._aps_scheduler.shutdown(wait=False)  # type: ignore[union-attr]
            self._aps_scheduler = None
        elif self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def run_now(self, user_id: str) -> None:
        """Manually trigger decay for a single user."""
        logger.info("DecayScheduler: running decay for user=%s", user_id)
        self._engine.run_for_user(user_id)

    def run_all_users(self) -> None:
        """Run decay for every user that has CLAIMED or PINNED docs."""
        users = self._store.list_active_users()
        logger.info("DecayScheduler: running decay for %d users", len(users))
        for uid in users:
            self._engine.run_for_user(uid)

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently active."""
        return self._running

    # ─── Simple backend (threading.Timer) ──────────────────────────────────────

    def _start_simple(self) -> None:
        """Start a repeating Timer loop."""
        self._schedule_next()

    def _schedule_next(self) -> None:
        """Schedule the next timer tick."""
        if not self._running:
            return
        self._timer = threading.Timer(
            self._interval_seconds,
            self._simple_tick,
        )
        self._timer.daemon = True
        self._timer.start()

    def _simple_tick(self) -> None:
        """Execute one round and reschedule."""
        if not self._running:
            return
        try:
            self.run_all_users()
        except Exception:
            logger.exception("DecayScheduler: error in simple tick")
        self._schedule_next()

    # ─── APScheduler backend ──────────────────────────────────────────────────

    def _start_apscheduler(self) -> None:
        """Start an APScheduler BackgroundScheduler."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError as exc:
            raise ImportError(
                "APScheduler backend requires the 'apscheduler' package. "
                "Install it with:  pip install 'langchain-rag-wiki[scheduler]'"
            ) from exc

        sched = BackgroundScheduler()
        sched.add_job(
            self.run_all_users,
            trigger=IntervalTrigger(seconds=self._interval_seconds),
            id="rag_wiki_decay",
            replace_existing=True,
        )
        sched.start()
        self._aps_scheduler = sched
