"""
GlobalDocStore — SQLite-backed registry of document-level metadata for
cross-user analytics (fetch totals, decay, flags).

Uses SQLAlchemy Core (not ORM).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite returns naive datetimes; treat as UTC for round-trip consistency."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _iso_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    dt = _ensure_utc(value)
    return dt.isoformat() if dt is not None else None


@dataclass
class GlobalDocRecord:
    doc_id: str
    source: str = ""
    department: str = "untagged"
    doc_title: str = ""
    doc_path: str = ""
    ingested_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    total_fetch_count: int = 0
    unique_users_fetched: int = 0
    chunk_count: int = 0
    doc_size_chars: int = 0
    tags: str = ""
    qdrant_ids: str = ""
    global_decay_score: float = 1.0
    is_flagged: bool = False
    flag_reason: str = ""


class GlobalDocStore:
    """
    SQLite-backed global document registry.

    Usage:
        store = GlobalDocStore("sqlite:///./rag_wiki.db")
        # or in-memory for tests:
        store = GlobalDocStore("sqlite:///:memory:")
    """

    def __init__(self, url: str = "sqlite:///./rag_wiki.db") -> None:
        self._engine = create_engine(url, echo=False)
        self._meta = MetaData()
        self._table = self._define_table()
        self._meta.create_all(self._engine)

    # ─── Schema ────────────────────────────────────────────────────────────────

    def _define_table(self) -> Table:
        return Table(
            "global_document_registry",
            self._meta,
            Column("doc_id", String, primary_key=True),
            Column("source", String, nullable=False, default=""),
            Column("department", String, nullable=False, default="untagged"),
            Column("doc_title", String, nullable=False, default=""),
            Column("doc_path", String, nullable=False, default=""),
            Column("ingested_at", DateTime, nullable=True),
            Column("last_updated_at", DateTime, nullable=True),
            Column("total_fetch_count", Integer, nullable=False, default=0),
            Column("unique_users_fetched", Integer, nullable=False, default=0),
            Column("chunk_count", Integer, nullable=False, default=0),
            Column("doc_size_chars", Integer, nullable=False, default=0),
            Column("tags", String, nullable=False, default=""),
            Column("qdrant_ids", String, nullable=False, default=""),
            Column("global_decay_score", Float, nullable=False, default=1.0),
            Column("is_flagged", Boolean, nullable=False, default=False),
            Column("flag_reason", String, nullable=False, default=""),
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────────

    def _row_to_record(self, row) -> GlobalDocRecord:
        return GlobalDocRecord(
            doc_id=row.doc_id,
            source=row.source,
            department=row.department,
            doc_title=row.doc_title,
            doc_path=row.doc_path,
            ingested_at=_ensure_utc(row.ingested_at),
            last_updated_at=_ensure_utc(row.last_updated_at),
            total_fetch_count=row.total_fetch_count,
            unique_users_fetched=row.unique_users_fetched,
            chunk_count=row.chunk_count,
            doc_size_chars=row.doc_size_chars,
            tags=row.tags,
            qdrant_ids=row.qdrant_ids,
            global_decay_score=row.global_decay_score,
            is_flagged=bool(row.is_flagged),
            flag_reason=row.flag_reason,
        )

    def _record_to_dict(self, record: GlobalDocRecord) -> dict:
        return {
            "doc_id": record.doc_id,
            "source": record.source,
            "department": record.department,
            "doc_title": record.doc_title,
            "doc_path": record.doc_path,
            "ingested_at": record.ingested_at,
            "last_updated_at": record.last_updated_at,
            "total_fetch_count": record.total_fetch_count,
            "unique_users_fetched": record.unique_users_fetched,
            "chunk_count": record.chunk_count,
            "doc_size_chars": record.doc_size_chars,
            "tags": record.tags,
            "qdrant_ids": record.qdrant_ids,
            "global_decay_score": record.global_decay_score,
            "is_flagged": record.is_flagged,
            "flag_reason": record.flag_reason,
        }

    # ─── API ─────────────────────────────────────────────────────────────────────

    def get(self, doc_id: str) -> Optional[GlobalDocRecord]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._table).where(self._table.c.doc_id == doc_id)
            ).first()
        return self._row_to_record(row) if row else None

    def upsert(self, record: GlobalDocRecord) -> None:
        data = self._record_to_dict(record)
        data["last_updated_at"] = _now()
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(self._table).where(self._table.c.doc_id == record.doc_id)
            ).first()

            if existing:
                conn.execute(
                    update(self._table)
                    .where(self._table.c.doc_id == record.doc_id)
                    .values(
                        **{
                            k: v
                            for k, v in data.items()
                            if k != "doc_id"
                        }
                    )
                )
            else:
                conn.execute(insert(self._table).values(**data))

    def delete(self, doc_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(delete(self._table).where(self._table.c.doc_id == doc_id))

    def list_all(self, limit: int = 500) -> list[GlobalDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table)
                .order_by(self._table.c.ingested_at.desc())
                .limit(limit)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_department(self, department: str) -> list[GlobalDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table)
                .where(self._table.c.department == department)
                .order_by(self._table.c.ingested_at.desc())
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_source(self, source: str) -> list[GlobalDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table)
                .where(self._table.c.source == source)
                .order_by(self._table.c.ingested_at.desc())
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_flagged(self) -> list[GlobalDocRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table)
                .where(self._table.c.is_flagged == True)  # noqa: E712
                .order_by(self._table.c.last_updated_at.desc())
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def increment_fetch(self, doc_id: str, user_id: str) -> None:
        with self._engine.begin() as conn:
            row = conn.execute(
                select(
                    self._table.c.total_fetch_count,
                    self._table.c.unique_users_fetched,
                ).where(self._table.c.doc_id == doc_id)
            ).first()
            if not row:
                return
            prev_total = row.total_fetch_count
            new_total = prev_total + 1
            if prev_total == 0:
                new_unique = row.unique_users_fetched + 1
            else:
                new_unique = row.unique_users_fetched
            conn.execute(
                update(self._table)
                .where(self._table.c.doc_id == doc_id)
                .values(
                    total_fetch_count=new_total,
                    unique_users_fetched=new_unique,
                    last_updated_at=_now(),
                )
            )

    def flag(self, doc_id: str, reason: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._table)
                .where(self._table.c.doc_id == doc_id)
                .values(
                    is_flagged=True,
                    flag_reason=reason,
                    last_updated_at=_now(),
                )
            )

    def unflag(self, doc_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._table)
                .where(self._table.c.doc_id == doc_id)
                .values(
                    is_flagged=False,
                    flag_reason="",
                    last_updated_at=_now(),
                )
            )

    def update_decay_score(self, doc_id: str, score: float) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._table)
                .where(self._table.c.doc_id == doc_id)
                .values(
                    global_decay_score=score,
                    last_updated_at=_now(),
                )
            )

    def update_doc_path(self, doc_id: str, path: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(self._table)
                .where(self._table.c.doc_id == doc_id)
                .values(
                    doc_path=path,
                    last_updated_at=_now(),
                )
            )

    def get_stats(self) -> dict:
        with self._engine.connect() as conn:
            total_documents = conn.execute(
                select(func.count()).select_from(self._table)
            ).scalar_one()

            total_chunks_raw = conn.execute(
                select(func.coalesce(func.sum(self._table.c.chunk_count), 0)).select_from(
                    self._table
                )
            ).scalar_one()

            flagged_count = conn.execute(
                select(func.count())
                .select_from(self._table)
                .where(self._table.c.is_flagged == True)  # noqa: E712
            ).scalar_one()

            avg_decay = conn.execute(
                select(func.avg(self._table.c.global_decay_score)).select_from(
                    self._table
                )
            ).scalar_one()

            dept_rows = conn.execute(
                select(self._table.c.department, func.count().label("n"))
                .group_by(self._table.c.department)
                .order_by(self._table.c.department)
            ).mappings().all()
            by_department = {r["department"]: r["n"] for r in dept_rows}

            src_rows = conn.execute(
                select(self._table.c.source, func.count().label("n"))
                .group_by(self._table.c.source)
                .order_by(self._table.c.source)
            ).mappings().all()
            by_source = {r["source"]: r["n"] for r in src_rows}

            oldest = conn.execute(
                select(func.min(self._table.c.ingested_at)).select_from(self._table)
            ).scalar_one()
            newest = conn.execute(
                select(func.max(self._table.c.ingested_at)).select_from(self._table)
            ).scalar_one()

        return {
            "total_documents": int(total_documents),
            "total_chunks": int(total_chunks_raw),
            "flagged_count": int(flagged_count),
            "average_decay_score": float(avg_decay) if avg_decay is not None else None,
            "by_department": by_department,
            "by_source": by_source,
            "oldest_ingested_at": _iso_utc(oldest),
            "newest_ingested_at": _iso_utc(newest),
        }

    def list_departments(self) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table.c.department)
                .distinct()
                .order_by(self._table.c.department)
            ).fetchall()
        return [r.department for r in rows]

    def list_sources(self) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._table.c.source).distinct().order_by(self._table.c.source)
            ).fetchall()
        return [r.source for r in rows]
