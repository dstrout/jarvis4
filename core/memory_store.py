"""
Memory Store — SQLite + vector embeddings for Jarvis long-term memory.

Stores atomic memories (preferences, habits, facts, personality observations)
with embeddings for semantic retrieval. Handles consolidation (merge/replace
duplicates) and decay (prune stale, low-value memories).

Embedding model loads lazily on first use to avoid startup delay.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("jarvis.memory_store")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str
    content: str
    category: str          # preference, habit, fact, personality
    importance: float      # 1-10
    access_count: int = 0
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


# ---------------------------------------------------------------------------
# Embedding service (lazy-loaded sentence-transformers)
# ---------------------------------------------------------------------------

_embedder = None
_embeddings_unavailable = False
_EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, fast, good quality
_EMBED_DIM = 384


class EmbeddingsUnavailable(RuntimeError):
    """Raised when semantic features are used without sentence-transformers."""


def embeddings_available() -> bool:
    """Whether semantic search is usable in this install.

    sentence-transformers is an optional extra (it pulls in torch, ~400MB).
    Without it the store still works as a plain SQLite memory store: memories
    are saved and listed, and search() falls back to keyword matching.
    """
    global _embeddings_unavailable
    if _embeddings_unavailable:
        return False
    if _embedder is not None:
        return True
    try:
        _get_embedder()
        return True
    except EmbeddingsUnavailable:
        return False


def _get_embedder():
    global _embedder, _embeddings_unavailable
    if _embedder is None:
        log.info("Loading embedding model: %s", _EMBED_MODEL)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            _embeddings_unavailable = True
            raise EmbeddingsUnavailable(
                "Semantic memory needs sentence-transformers. "
                "Install it with: pip install -r requirements-memory.txt. "
                "Without it, memories are still stored and keyword-searchable."
            ) from exc
        _embedder = SentenceTransformer(_EMBED_MODEL)
        log.info("Embedding model loaded")
    return _embedder


def _maybe_embed(text: str):
    """Embed text, or return None when embeddings are unavailable."""
    if not embeddings_available():
        return None
    return embed_text(text)


def embed_text(text: str) -> np.ndarray:
    """Compute a normalized embedding vector for text."""
    model = _get_embedder()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch embed multiple texts. Returns (N, dim) array."""
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Memory Store (SQLite backend)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 5.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    embedding BLOB
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
"""


class MemoryStore:
    """Persistent memory store with semantic search."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "memory.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()
        log.info("MemoryStore opened: %s (%d memories)", self._db_path, self.count())

    # --- Core CRUD ---

    def add(self, content: str, category: str, importance: float = 5.0) -> Memory:
        """Add a new memory. Computes embedding automatically."""
        now = time.time()
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            category=category,
            importance=max(1.0, min(10.0, importance)),
            created_at=now,
            last_accessed_at=now,
            embedding=_maybe_embed(content),
        )
        self._insert(mem)
        log.info("Added memory [%s] %.1f: %s", category, importance, content[:80])
        return mem

    def update(self, memory_id: str, content: str, importance: float | None = None) -> bool:
        """Update a memory's content and re-embed."""
        row = self._conn.execute(
            "SELECT importance FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return False

        new_importance = importance if importance is not None else row[0]
        embedding = _maybe_embed(content)
        self._conn.execute(
            "UPDATE memories SET content=?, importance=?, embedding=?, last_accessed_at=? WHERE id=?",
            (
                content,
                new_importance,
                embedding.tobytes() if embedding is not None else None,
                time.time(),
                memory_id,
            ),
        )
        self._conn.commit()
        log.info("Updated memory %s: %s", memory_id[:8], content[:80])
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get(self, memory_id: str) -> Memory | None:
        """Get a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_all(self, category: str | None = None) -> list[Memory]:
        """Get all memories, optionally filtered by category."""
        if category:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY importance DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY importance DESC"
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # --- Semantic search ---

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        recency_weight: float = 0.1,
        importance_weight: float = 0.2,
        relevance_weight: float = 0.7,
    ) -> list[tuple[Memory, float]]:
        """
        Retrieve top-K memories scored by weighted combination of:
        - relevance (cosine similarity to query)
        - importance (normalized 0-1)
        - recency (exponential decay over hours)

        Returns list of (Memory, score) tuples, highest first.
        """
        all_memories = self.get_all(category)
        if not all_memories:
            return []

        semantic = embeddings_available()
        query_vec = embed_text(query) if semantic else None
        query_terms = {t for t in query.lower().split() if len(t) > 2}

        scored = []
        now = time.time()
        for mem in all_memories:
            if semantic:
                if mem.embedding is None:
                    continue
                relevance = cosine_similarity(query_vec, mem.embedding)
            else:
                # Keyword fallback: fraction of query terms present in content
                if not query_terms:
                    relevance = 0.0
                else:
                    content_lower = mem.content.lower()
                    hits = sum(1 for t in query_terms if t in content_lower)
                    relevance = hits / len(query_terms)

            importance_norm = mem.importance / 10.0
            age_hours = max((now - mem.last_accessed_at) / 3600, 0.01)
            recency = np.exp(-0.01 * age_hours)  # Half-life ~70 hours

            score = (
                relevance_weight * relevance
                + importance_weight * importance_norm
                + recency_weight * recency
            )
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Touch accessed memories
        top = scored[:top_k]
        for mem, _ in top:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                (now, mem.id),
            )
        if top:
            self._conn.commit()

        return top

    def find_similar(
        self, content: str, threshold: float = 0.80
    ) -> list[tuple[Memory, float]]:
        """Find memories with cosine similarity above threshold.

        Returns [] when embeddings are unavailable, so consolidation simply
        adds new memories instead of merging near-duplicates.
        """
        if not embeddings_available():
            return []
        vec = embed_text(content)
        results = []
        for mem in self.get_all():
            if mem.embedding is None:
                continue
            sim = cosine_similarity(vec, mem.embedding)
            if sim >= threshold:
                results.append((mem, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # --- Consolidation ---

    def consolidate(
        self, new_content: str, category: str, importance: float = 5.0
    ) -> tuple[str, Memory]:
        """
        Add-or-merge a memory. If a similar memory exists (>0.85 similarity),
        returns ("updated", merged_memory). Otherwise ("added", new_memory).

        This is the primary entry point for the extraction pipeline.
        """
        similar = self.find_similar(new_content, threshold=0.85)

        if similar:
            existing, sim = similar[0]
            # Very high similarity (>0.95) = near-duplicate, just bump importance
            if sim > 0.95:
                new_importance = max(existing.importance, importance)
                if new_importance != existing.importance:
                    self.update(existing.id, existing.content, new_importance)
                    existing.importance = new_importance
                log.info("Duplicate memory (sim=%.2f), bumped importance", sim)
                return "updated", existing
            # Moderate similarity = related, merge content
            merged = f"{existing.content} — also: {new_content}"
            if len(merged) > 500:
                # If too long, just replace with newer
                merged = new_content
            new_importance = max(existing.importance, importance)
            self.update(existing.id, merged, new_importance)
            existing.content = merged
            existing.importance = new_importance
            log.info(
                "Consolidated memory (sim=%.2f): %s", sim, merged[:80]
            )
            return "updated", existing
        else:
            mem = self.add(new_content, category, importance)
            return "added", mem

    # --- Maintenance ---

    def prune(self, max_age_days: int = 90, min_importance: float = 3.0) -> int:
        """Remove old, low-importance, never-accessed memories."""
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE created_at < ? AND importance < ? AND access_count = 0",
            (cutoff, min_importance),
        )
        self._conn.commit()
        if cursor.rowcount:
            log.info("Pruned %d stale memories", cursor.rowcount)
        return cursor.rowcount

    def get_stats(self) -> dict:
        """Return summary statistics."""
        rows = self._conn.execute(
            "SELECT category, COUNT(*), AVG(importance), AVG(access_count) FROM memories GROUP BY category"
        ).fetchall()
        return {
            "total": self.count(),
            "categories": {
                row[0]: {"count": row[1], "avg_importance": round(row[2], 1), "avg_accesses": round(row[3], 1)}
                for row in rows
            },
        }

    # --- Internal ---

    def _insert(self, mem: Memory):
        self._conn.execute(
            "INSERT INTO memories (id, content, category, importance, access_count, created_at, last_accessed_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mem.id,
                mem.content,
                mem.category,
                mem.importance,
                mem.access_count,
                mem.created_at,
                mem.last_accessed_at,
                mem.embedding.tobytes() if mem.embedding is not None else None,
            ),
        )
        self._conn.commit()

    def _row_to_memory(self, row: tuple) -> Memory:
        embedding = None
        if row[7]:
            embedding = np.frombuffer(row[7], dtype=np.float32).copy()
        return Memory(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            access_count=row[4],
            created_at=row[5],
            last_accessed_at=row[6],
            embedding=embedding,
        )

    def close(self):
        self._conn.close()
