"""
Tests for GlobalDocStore.
"""

import pytest
from datetime import datetime, timedelta, timezone

from rag_wiki.storage.global_store import GlobalDocRecord, GlobalDocStore


def make_global_record(**overrides) -> GlobalDocRecord:
    now = datetime.now(timezone.utc)
    defaults = dict(
        doc_id="doc-1",
        source="slack",
        department="engineering",
        doc_title="Test",
        doc_path="",
        ingested_at=now,
        last_updated_at=now,
        total_fetch_count=0,
        unique_users_fetched=0,
        chunk_count=0,
        doc_size_chars=0,
        tags="",
        qdrant_ids="",
        global_decay_score=1.0,
        is_flagged=False,
        flag_reason="",
    )
    defaults.update(overrides)
    return GlobalDocRecord(**defaults)


@pytest.fixture
def store():
    return GlobalDocStore("sqlite:///:memory:")


def test_upsert_and_get(store):
    t0 = datetime.now(timezone.utc)
    r = make_global_record(
        doc_id="d-1",
        source="notion",
        department="marketing",
        doc_title="Title",
        doc_path="/kb/x.md",
        ingested_at=t0,
        total_fetch_count=3,
        unique_users_fetched=2,
        chunk_count=10,
        doc_size_chars=5000,
        tags="a,b",
        qdrant_ids="1,2,3",
        global_decay_score=0.85,
        is_flagged=True,
        flag_reason="review",
    )
    store.upsert(r)
    fetched = store.get("d-1")
    assert fetched is not None
    assert fetched.doc_id == "d-1"
    assert fetched.source == "notion"
    assert fetched.department == "marketing"
    assert fetched.doc_title == "Title"
    assert fetched.doc_path == "/kb/x.md"
    assert fetched.ingested_at == t0
    assert fetched.last_updated_at is not None
    assert fetched.total_fetch_count == 3
    assert fetched.unique_users_fetched == 2
    assert fetched.chunk_count == 10
    assert fetched.doc_size_chars == 5000
    assert fetched.tags == "a,b"
    assert fetched.qdrant_ids == "1,2,3"
    assert fetched.global_decay_score == 0.85
    assert fetched.is_flagged is True
    assert fetched.flag_reason == "review"


def test_get_missing(store):
    assert store.get("no-such-doc") is None


def test_delete(store):
    store.upsert(make_global_record(doc_id="del-me"))
    store.delete("del-me")
    assert store.get("del-me") is None


def test_list_all(store):
    store.upsert(make_global_record(doc_id="a"))
    store.upsert(make_global_record(doc_id="b"))
    store.upsert(make_global_record(doc_id="c"))
    rows = store.list_all()
    assert len(rows) == 3
    assert {r.doc_id for r in rows} == {"a", "b", "c"}


def test_list_by_department(store):
    store.upsert(make_global_record(doc_id="e1", department="engineering"))
    store.upsert(make_global_record(doc_id="e2", department="engineering"))
    store.upsert(make_global_record(doc_id="m1", department="marketing"))
    eng = store.list_by_department("engineering")
    assert len(eng) == 2
    assert {r.doc_id for r in eng} == {"e1", "e2"}


def test_list_by_source(store):
    store.upsert(make_global_record(doc_id="s1", source="slack"))
    store.upsert(make_global_record(doc_id="n1", source="notion"))
    slack = store.list_by_source("slack")
    assert len(slack) == 1
    assert slack[0].doc_id == "s1"


def test_list_flagged(store):
    store.upsert(make_global_record(doc_id="ok", is_flagged=False))
    store.upsert(make_global_record(doc_id="bad", is_flagged=True, flag_reason="spam"))
    flagged = store.list_flagged()
    assert len(flagged) == 1
    assert flagged[0].doc_id == "bad"


def test_flag_unflag(store):
    store.upsert(make_global_record(doc_id="f1", is_flagged=False))
    store.flag("f1", "reason")
    r = store.get("f1")
    assert r.is_flagged is True
    assert r.flag_reason == "reason"
    store.unflag("f1")
    r2 = store.get("f1")
    assert r2.is_flagged is False
    assert r2.flag_reason == ""


def test_increment_fetch(store):
    store.upsert(make_global_record(doc_id="inc", total_fetch_count=0))
    store.increment_fetch("inc", "user-1")
    store.increment_fetch("inc", "user-2")
    r = store.get("inc")
    assert r.total_fetch_count == 2


def test_get_stats(store):
    base = datetime.now(timezone.utc)
    t1 = base
    t2 = base + timedelta(hours=1)
    t3 = base + timedelta(hours=2)
    store.upsert(
        make_global_record(
            doc_id="st-a",
            department="engineering",
            source="slack",
            chunk_count=5,
            global_decay_score=1.0,
            is_flagged=False,
            ingested_at=t1,
        )
    )
    store.upsert(
        make_global_record(
            doc_id="st-b",
            department="engineering",
            source="notion",
            chunk_count=3,
            global_decay_score=0.5,
            is_flagged=False,
            ingested_at=t2,
        )
    )
    store.upsert(
        make_global_record(
            doc_id="st-c",
            department="marketing",
            source="slack",
            chunk_count=2,
            global_decay_score=0.5,
            is_flagged=True,
            flag_reason="x",
            ingested_at=t3,
        )
    )
    stats = store.get_stats()
    assert stats["total_documents"] == 3
    assert stats["total_chunks"] == 10
    assert stats["flagged_count"] == 1
    assert stats["average_decay_score"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert stats["by_department"] == {"engineering": 2, "marketing": 1}
    assert stats["by_source"] == {"notion": 1, "slack": 2}
    assert stats["oldest_ingested_at"] == t1.isoformat()
    assert stats["newest_ingested_at"] == t3.isoformat()


def test_update_decay_score(store):
    store.upsert(make_global_record(doc_id="dec", global_decay_score=1.0))
    store.update_decay_score("dec", 0.3)
    assert store.get("dec").global_decay_score == 0.3


def test_update_doc_path(store):
    store.upsert(make_global_record(doc_id="path", doc_path=""))
    store.update_doc_path("path", "/new/path.md")
    assert store.get("path").doc_path == "/new/path.md"


def test_list_departments(store):
    store.upsert(make_global_record(doc_id="d1", department="zebra"))
    store.upsert(make_global_record(doc_id="d2", department="alpha"))
    store.upsert(make_global_record(doc_id="d3", department="alpha"))
    depts = store.list_departments()
    assert depts == ["alpha", "zebra"]


def test_list_sources(store):
    store.upsert(make_global_record(doc_id="s1", source="notion"))
    store.upsert(make_global_record(doc_id="s2", source="file"))
    store.upsert(make_global_record(doc_id="s3", source="file"))
    sources = store.list_sources()
    assert sources == ["file", "notion"]
