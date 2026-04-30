"""
SQLiteStateStore — zero-infrastructure backend for local dev and small deployments.
Uses SQLAlchemy Core (not ORM) for minimal overhead and maximum portability.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    create_engine, MetaData, Table, select, insert, update, delete
)

from .base import StateStore, UserDocRecord, DocumentState


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteStateStore(StateStore):
    """
    SQLite-backed state store.

    Usage:
        store = SQLiteStateStore("sqlite:///./rag_wiki.db")
        # or in-memory for tests:
        store = SQLiteStateStore("sqlite:///:memory:")
    """

    def __init__(self, url: str = "sqlite:///./rag_wiki.db"):
        self._engine = create_engine(url, echo=False)
        self._meta   = MetaData()
        self._table  = self._define_table()
        self._meta.create_all(self._engine)

    # ─── Schema ────────────────────────────────────────────────────────────────

    def _define_table(self) -> Table:
        return Table(
            "user_document_state",
            self._meta,
            Column("user_id",              String,  primary_key=True),
            Column("doc_id",               String,  primary_key=True),
            Column("doc_title",            String,  nullable=False, default=""),
            Column("doc_path",             String,  nullable=False, default=""),
            Column("user_state",           String,  nullable=False, default=DocumentState.SURFACED),
            Column("fetch_count",          Integer, nullable=False, default=0),
            Column("last_fetched_at",      DateTime),
            Column("suggestion_sent",      Boolean, nullable=False, default=False),
            Column("decay_score",          Float,   nullable=False, default=1.0),
            Column("explicit_signal",      Float,   nullable=False, default=0.0),
            Column("full_content",         Text),
            Column("cached_at",            DateTime),
            Column("pinned_at",            DateTime),
            Column("demoted_at",           DateTime),
            Column("no_resiluggest_until", DateTime),
            Column("next_suggest_at",      Integer, nullable=False, default=0),
            Column("queries_missed",       Integer, nullable=False, default=0),
            Column("cache_miss_streak",    Integer, nullable=False, default=0),
        )

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _row_to_record(self, row) -> UserDocRecord:
        return UserDocRecord(
            user_id              = row.user_id,
            doc_id               = row.doc_id,
            doc_title            = row.doc_title,
            doc_path             = row.doc_path,
            user_state           = DocumentState(row.user_state),
            fetch_count          = row.fetch_count,
            last_fetched_at      = row.last_fetched_at,
            suggestion_sent      = row.suggestion_sent,
            decay_score          = row.decay_score,
            explicit_signal      = row.explicit_signal,
            full_content         = row.full_content,
            cached_at            = row.cached_at,
            pinned_at            = row.pinned_at,
            demoted_at           = row.demoted_at,
            no_resiluggest_until = row.no_resiluggest_until,
            next_suggest_at      = row.next_suggest_at if hasattr(row, "next_suggest_at") else 0,
            queries_missed       = row.queries_missed if hasattr(row, "queries_missed") else 0,
            cache_miss_streak    = row.cache_miss_streak if hasattr(row, "cache_miss_streak") else 0,
        )

    def _record_to_dict(self, record: UserDocRecord) -> dict:
        return {
            "user_id":              record.user_id,
            "doc_id":               record.doc_id,
            "doc_title":            record.doc_title,
            "doc_path":             record.doc_path,
            "user_state":           record.user_state.value,
            "fetch_count":          record.fetch_count,
            "last_fetched_at":      record.last_fetched_at,
            "suggestion_sent":      record.suggestion_sent,
            "decay_score":          record.decay_score,
            "explicit_signal":      record.explicit_signal,
            "full_content":         record.full_content,
            "cached_at":            record.cached_at,
            "pinned_at":            record.pinned_at,
            "demoted_at":           record.demoted_at,
            "no_resiluggest_until": record.no_resiluggest_until,
            "next_suggest_at":      record.next_suggest_at,
            "queries_missed":       record.queries_missed,
            "cache_miss_streak":    record.cache_miss_streak,
        }

    # ─── StateStore interface ──────────────────────────────────────────────────

    def get(self, user_id: str, doc_id: str) -> Optional[UserDocRecord]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._table).where(
                    self._table.c.user_id == user_id,
                    self._table.c.doc_id  == doc_id,
                )
            ).first()
        return self._row_to_record(row) if row else None

    def upsert(self, record: UserDocRecord) -> None:
        data = self._record_to_dict(record)
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(self._table).where(
                    self._table.c.user_id == record.user_id,
                    self._table.c.doc_id  == record.doc_id,
                )
            ).first()

            if existing:
                conn.execute(
                    update(self._table)
                    .where(
                        self._table.c.user_id == record.user_id,
                        self._table.c.doc_id  == record.doc_id,
                    )
                    .values(**{k: v for k, v in data.items()
                               if k not in ("user_id", "doc_id")})
                )
            else:
                conn.execute(insert(self._table).values(**data))

    def list_claimed(self, user_id: str) -> list[UserDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table).where(
                    self._table.c.user_id    == user_id,
                    self._table.c.user_state == DocumentState.CLAIMED.value,
                )
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_pinned(self, user_id: str) -> list[UserDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table).where(
                    self._table.c.user_id    == user_id,
                    self._table.c.user_state == DocumentState.PINNED.value,
                )
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_surfaced(self, user_id: str) -> list[UserDocRecord]:
        """All SURFACED and SUGGESTED docs — used for miss tracking."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table).where(
                    self._table.c.user_id == user_id,
                    self._table.c.user_state.in_([
                        DocumentState.SURFACED.value,
                        DocumentState.SUGGESTED.value,
                    ]),
                )
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_for_decay(self, user_id: str) -> list[UserDocRecord]:
        """Returns CLAIMED + PINNED records for the decay engine."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table).where(
                    self._table.c.user_id == user_id,
                    self._table.c.user_state.in_([
                        DocumentState.CLAIMED.value,
                        DocumentState.PINNED.value,
                    ]),
                )
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete(self, user_id: str, doc_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(
                    self._table.c.user_id == user_id,
                    self._table.c.doc_id  == doc_id,
                )
            )

    def list_active_users(self) -> list[str]:
        """Return distinct user_ids that have at least one CLAIMED or PINNED doc."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table.c.user_id)
                .where(
                    self._table.c.user_state.in_([
                        DocumentState.CLAIMED.value,
                        DocumentState.PINNED.value,
                    ])
                )
                .distinct()
            ).fetchall()
        return sorted(row.user_id for row in rows)