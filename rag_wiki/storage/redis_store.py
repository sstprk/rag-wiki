"""
RedisStateStore — Redis-backed state storage for distributed deployments.

Each ``UserDocRecord`` is stored as a Redis hash under the key pattern
``hkb:{user_id}:{doc_id}``.  State index sets are maintained for fast
lookups: ``hkb:{user_id}:state:{state_value}`` stores a Redis SET of
``doc_id`` values in that state.

.. note::
   Requires ``redis-py`` (``pip install redis>=4.0``).
   If ``redis`` is not installed, importing this module raises an
   ``ImportError`` with a helpful message.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    import redis
except ImportError as exc:
    raise ImportError(
        "RedisStateStore requires the 'redis' package. "
        "Install it with:  pip install 'langchain-rag-wiki[redis]'"
    ) from exc

from rag_wiki.storage.base import DocumentState, StateStore, UserDocRecord


def _key(user_id: str, doc_id: str) -> str:
    """Hash key for a single user-doc record."""
    return f"hkb:{user_id}:{doc_id}"


def _state_set_key(user_id: str, state: DocumentState) -> str:
    """Set key for fast state lookups."""
    return f"hkb:{user_id}:state:{state.value}"


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO format string, or None."""
    return dt.isoformat() if dt else None


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    """Deserialize an ISO format string to datetime, or None."""
    if s is None or s == "" or s == "None":
        return None
    return datetime.fromisoformat(s)


class RedisStateStore(StateStore):
    """
    Redis-backed state store for production / distributed use.

    Usage::

        import redis
        from rag_wiki.storage.redis_store import RedisStateStore

        client = redis.Redis(host="localhost", port=6379, db=0)
        store  = RedisStateStore(client)
    """

    def __init__(self, client: redis.Redis) -> None:
        self._r = client

    # ─── Serialization helpers ─────────────────────────────────────────────────

    def _record_to_hash(self, record: UserDocRecord) -> dict[str, str]:
        """Convert a UserDocRecord to a flat dict of strings for Redis HSET."""
        return {
            "user_id":              record.user_id,
            "doc_id":               record.doc_id,
            "doc_title":            record.doc_title,
            "doc_path":             record.doc_path,
            "user_state":           record.user_state.value,
            "fetch_count":          str(record.fetch_count),
            "last_fetched_at":      _dt_to_str(record.last_fetched_at) or "",
            "suggestion_sent":      "1" if record.suggestion_sent else "0",
            "decay_score":          str(record.decay_score),
            "explicit_signal":      str(record.explicit_signal),
            "full_content":         record.full_content or "",
            "cached_at":            _dt_to_str(record.cached_at) or "",
            "pinned_at":            _dt_to_str(record.pinned_at) or "",
            "demoted_at":           _dt_to_str(record.demoted_at) or "",
            "no_resiluggest_until": _dt_to_str(record.no_resiluggest_until) or "",
            "next_suggest_at":      str(record.next_suggest_at),
            "queries_missed":       str(record.queries_missed),
            "cache_miss_streak":    str(record.cache_miss_streak),
        }

    def _hash_to_record(self, data: dict[bytes | str, bytes | str]) -> UserDocRecord:
        """Convert a Redis hash (bytes or str values) back to a UserDocRecord."""
        def _v(key: str) -> str:
            val = data.get(key) or data.get(key.encode())
            if isinstance(val, bytes):
                return val.decode()
            return val or ""

        return UserDocRecord(
            user_id              = _v("user_id"),
            doc_id               = _v("doc_id"),
            doc_title            = _v("doc_title"),
            doc_path             = _v("doc_path"),
            user_state           = DocumentState(_v("user_state")),
            fetch_count          = int(_v("fetch_count") or "0"),
            last_fetched_at      = _str_to_dt(_v("last_fetched_at")),
            suggestion_sent      = _v("suggestion_sent") == "1",
            decay_score          = float(_v("decay_score") or "1.0"),
            explicit_signal      = float(_v("explicit_signal") or "0.0"),
            full_content         = _v("full_content") or None,
            cached_at            = _str_to_dt(_v("cached_at")),
            pinned_at            = _str_to_dt(_v("pinned_at")),
            demoted_at           = _str_to_dt(_v("demoted_at")),
            no_resiluggest_until = _str_to_dt(_v("no_resiluggest_until")),
            next_suggest_at      = int(_v("next_suggest_at") or "0"),
            queries_missed       = int(_v("queries_missed") or "0"),
            cache_miss_streak    = int(_v("cache_miss_streak") or "0"),
        )

    # ─── Index management ─────────────────────────────────────────────────────

    _ACTIVE_USERS_KEY = "hkb:active_users"
    _ACTIVE_STATES = frozenset({DocumentState.CLAIMED, DocumentState.PINNED})

    def _update_state_index(
        self,
        user_id: str,
        doc_id: str,
        old_state: Optional[DocumentState],
        new_state: DocumentState,
    ) -> None:
        """Move doc_id between state index sets and maintain active_users."""
        pipe = self._r.pipeline()
        if old_state is not None:
            pipe.srem(_state_set_key(user_id, old_state), doc_id)
        pipe.sadd(_state_set_key(user_id, new_state), doc_id)

        # Maintain global active_users set
        if new_state in self._ACTIVE_STATES:
            pipe.sadd(self._ACTIVE_USERS_KEY, user_id)
        pipe.execute()

        # If leaving active state, check if user still has active docs
        if (
            old_state in self._ACTIVE_STATES
            and new_state not in self._ACTIVE_STATES
        ):
            self._refresh_user_active_status(user_id)

    def _refresh_user_active_status(self, user_id: str) -> None:
        """Remove user from active set if they have no CLAIMED/PINNED docs."""
        claimed_count = self._r.scard(
            _state_set_key(user_id, DocumentState.CLAIMED)
        )
        pinned_count = self._r.scard(
            _state_set_key(user_id, DocumentState.PINNED)
        )
        if claimed_count == 0 and pinned_count == 0:
            self._r.srem(self._ACTIVE_USERS_KEY, user_id)

    # ─── StateStore interface ──────────────────────────────────────────────────

    def get(self, user_id: str, doc_id: str) -> Optional[UserDocRecord]:
        """Return the record for this user+doc pair, or ``None``."""
        data = self._r.hgetall(_key(user_id, doc_id))
        if not data:
            return None
        return self._hash_to_record(data)

    def upsert(self, record: UserDocRecord) -> None:
        """Insert or update a record."""
        k = _key(record.user_id, record.doc_id)
        old_data = self._r.hgetall(k)
        old_state: Optional[DocumentState] = None
        if old_data:
            raw = old_data.get("user_state") or old_data.get(b"user_state")
            if isinstance(raw, bytes):
                raw = raw.decode()
            if raw:
                old_state = DocumentState(raw)

        self._r.hset(k, mapping=self._record_to_hash(record))
        self._update_state_index(
            record.user_id, record.doc_id, old_state, record.user_state
        )

    def list_claimed(self, user_id: str) -> list[UserDocRecord]:
        """All CLAIMED docs for a user."""
        return self._list_by_state(user_id, DocumentState.CLAIMED)

    def list_pinned(self, user_id: str) -> list[UserDocRecord]:
        """All PINNED docs for a user."""
        return self._list_by_state(user_id, DocumentState.PINNED)

    def list_for_decay(self, user_id: str) -> list[UserDocRecord]:
        """CLAIMED + PINNED records eligible for decay scoring."""
        return self._list_by_state(user_id, DocumentState.CLAIMED) + \
               self._list_by_state(user_id, DocumentState.PINNED)

    def list_surfaced(self, user_id: str) -> list[UserDocRecord]:
        """All SURFACED and SUGGESTED docs — used for miss tracking."""
        return (
            self._list_by_state(user_id, DocumentState.SURFACED) +
            self._list_by_state(user_id, DocumentState.SUGGESTED)
        )

    def delete(self, user_id: str, doc_id: str) -> None:
        """Remove a record entirely."""
        k = _key(user_id, doc_id)
        data = self._r.hgetall(k)
        if data:
            record = self._hash_to_record(data)
            self._r.srem(_state_set_key(user_id, record.user_state), doc_id)
        self._r.delete(k)
        self._refresh_user_active_status(user_id)

    def list_active_users(self) -> list[str]:
        """Return distinct user_ids with at least one CLAIMED or PINNED doc.

        Uses the ``hkb:active_users`` global set for O(1) lookups
        instead of scanning the entire keyspace.
        """
        members = self._r.smembers(self._ACTIVE_USERS_KEY)
        users: list[str] = []
        for m in members:
            if isinstance(m, bytes):
                m = m.decode()
            users.append(m)
        return sorted(users)

    # ─── Private helpers ───────────────────────────────────────────────────────

    def _list_by_state(
        self, user_id: str, state: DocumentState
    ) -> list[UserDocRecord]:
        """Fetch all records for a user in a given state via the index set."""
        set_key = _state_set_key(user_id, state)
        doc_ids = self._r.smembers(set_key)
        records: list[UserDocRecord] = []
        for did in doc_ids:
            if isinstance(did, bytes):
                did = did.decode()
            data = self._r.hgetall(_key(user_id, did))
            if data:
                records.append(self._hash_to_record(data))
        return records

