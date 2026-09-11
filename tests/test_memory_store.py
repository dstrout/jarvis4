#!/usr/bin/env python3
"""
Tests for the SQLite memory store.

Covers CRUD, pruning and stats against a temporary database, plus the
degraded path taken when sentence-transformers is not installed (semantic
search is an optional extra, so the store must stay usable without it).

Usage:
    python -m pytest tests/test_memory_store.py
"""

import pytest

from core import memory_store
from core.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """A MemoryStore backed by a throwaway database."""
    s = MemoryStore(db_path=tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def no_embeddings(monkeypatch):
    """Simulate an install without sentence-transformers."""
    monkeypatch.setattr(memory_store, "embeddings_available", lambda: False)


# --- CRUD ---------------------------------------------------------------


def test_add_returns_memory_and_persists(store):
    mem = store.add("prefers dim lights after 9pm", "preference", importance=7.0)

    assert mem.id
    assert mem.content == "prefers dim lights after 9pm"
    assert mem.category == "preference"
    assert mem.importance == 7.0
    assert store.count() == 1
    assert store.get(mem.id).content == mem.content


def test_importance_is_clamped_to_range(store):
    assert store.add("way too important", "fact", importance=99.0).importance == 10.0
    assert store.add("not important at all", "fact", importance=-5.0).importance == 1.0


def test_get_missing_returns_none(store):
    assert store.get("does-not-exist") is None


def test_get_all_filters_by_category(store):
    store.add("works from a home office", "fact")
    store.add("checks email first thing", "habit")
    store.add("has two aquariums", "fact")

    assert len(store.get_all()) == 3
    assert len(store.get_all(category="fact")) == 2
    assert len(store.get_all(category="habit")) == 1
    assert store.get_all(category="nonexistent") == []


def test_update_changes_content_and_importance(store):
    mem = store.add("likes tea", "preference", importance=4.0)

    assert store.update(mem.id, "likes strong black tea", importance=8.0) is True

    updated = store.get(mem.id)
    assert updated.content == "likes strong black tea"
    assert updated.importance == 8.0


def test_update_keeps_importance_when_omitted(store):
    mem = store.add("likes tea", "preference", importance=4.0)
    store.update(mem.id, "likes green tea")
    assert store.get(mem.id).importance == 4.0


def test_update_missing_returns_false(store):
    assert store.update("does-not-exist", "anything") is False


def test_delete_removes_memory(store):
    mem = store.add("temporary note", "fact")
    assert store.delete(mem.id) is True
    assert store.get(mem.id) is None
    assert store.count() == 0


def test_delete_missing_returns_false(store):
    assert store.delete("does-not-exist") is False


# --- Persistence --------------------------------------------------------


def test_memories_survive_reopen(tmp_path):
    db = tmp_path / "memory.db"
    first = MemoryStore(db_path=db)
    first.add("survives a restart", "fact", importance=6.0)
    first.close()

    second = MemoryStore(db_path=db)
    try:
        assert second.count() == 1
        assert second.get_all()[0].content == "survives a restart"
    finally:
        second.close()


# --- Pruning and stats --------------------------------------------------


def test_prune_drops_old_low_importance_memories(store):
    keep_important = store.add("critical fact", "fact", importance=9.0)
    keep_recent = store.add("recent trivia", "fact", importance=1.0)
    drop = store.add("stale trivia", "fact", importance=1.0)

    # Age the low-importance memory past the cutoff
    ancient = 0.0
    store._conn.execute(
        "UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?",
        (ancient, ancient, drop.id),
    )
    store._conn.commit()

    removed = store.prune(max_age_days=90, min_importance=3.0)

    assert removed == 1
    remaining = {m.id for m in store.get_all()}
    assert drop.id not in remaining
    assert keep_important.id in remaining
    assert keep_recent.id in remaining


def test_get_stats_reports_totals_by_category(store):
    store.add("a preference", "preference", importance=5.0)
    store.add("a habit", "habit", importance=5.0)
    store.add("another habit", "habit", importance=5.0)

    stats = store.get_stats()

    assert stats["total"] == 3
    assert stats["categories"]["habit"]["count"] == 2
    assert stats["categories"]["preference"]["count"] == 1
    assert stats["categories"]["habit"]["avg_importance"] == 5.0


def test_age_hours_grows_with_age(store):
    mem = store.add("some fact", "fact")
    assert mem.age_hours() < 1.0

    mem.created_at -= 7200  # two hours back
    assert 1.9 < mem.age_hours() < 2.1


# --- Search -------------------------------------------------------------


def test_search_on_empty_store_returns_empty(store):
    assert store.search("anything") == []


def test_search_ranks_and_limits_results(store):
    store.add("prefers dim lights in the evening", "preference", importance=8.0)
    store.add("drives an electric car", "fact", importance=5.0)
    store.add("enjoys black coffee", "preference", importance=5.0)

    results = store.search("what lighting does the user like", top_k=2)

    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert "lights" in results[0][0].content


def test_search_increments_access_count(store):
    mem = store.add("prefers dim lights", "preference")
    assert store.get(mem.id).access_count == 0

    store.search("lights", top_k=1)

    assert store.get(mem.id).access_count == 1


# --- Degraded mode: no sentence-transformers ----------------------------


def test_add_works_without_embeddings(store, no_embeddings):
    mem = store.add("stored without an embedding", "fact")

    assert store.count() == 1
    assert store.get(mem.id).embedding is None


def test_search_falls_back_to_keywords(store, no_embeddings):
    store.add("prefers dim lights in the evening", "preference", importance=5.0)
    store.add("drives an electric car", "fact", importance=5.0)

    results = store.search("dim lights", top_k=2)

    assert results
    assert "lights" in results[0][0].content


def test_find_similar_returns_empty_without_embeddings(store, no_embeddings):
    store.add("prefers dim lights", "preference")
    assert store.find_similar("prefers dim lighting") == []


def test_update_works_without_embeddings(store, no_embeddings):
    mem = store.add("original wording", "fact")
    assert store.update(mem.id, "revised wording") is True
    assert store.get(mem.id).content == "revised wording"
    assert store.get(mem.id).embedding is None
