"""Stockage SQLite : messages, chunks, index plein texte FTS5 et index vectoriels sqlite-vec.

Une table `vec0` est créée par couple (fournisseur, modèle, dimension) : les vecteurs
de deux espaces d'embedding ne sont jamais comparés entre eux.
"""

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from zeroctx.providers.base import Role


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    message_id: int
    role: str
    content: str
    score: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    topic_id TEXT,
    generation_id TEXT,
    status TEXT NOT NULL DEFAULT 'complete',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_message ON chunks(message_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _vec_table(embedder_key: str) -> str:
    return "vec_" + "".join(c if c.isalnum() or c == "_" else "_" for c in embedder_key)


def _fts_query(text: str) -> str | None:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return None
    # Chaque token est cité pour éviter toute syntaxe FTS ; combinaison en OR.
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.executescript(SCHEMA)
        self._vec_dims: dict[str, int] = {}

    def close(self) -> None:
        self.conn.close()

    # ----- messages et chunks -------------------------------------------------

    def add_message(
        self,
        role: Role,
        content: str,
        topic_id: str | None,
        generation_id: str | None = None,
        status: str = "complete",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(role, content, topic_id, generation_id, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (role, content, topic_id, generation_id, status, time.time()),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def add_chunks(self, message_id: int, role: str, chunks: list[str]) -> list[int]:
        ids: list[int] = []
        for position, content in enumerate(chunks):
            cur = self.conn.execute(
                "INSERT INTO chunks(message_id, position, role, content) VALUES (?, ?, ?, ?)",
                (message_id, position, role, content),
            )
            assert cur.lastrowid is not None
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                (cur.lastrowid, content),
            )
            ids.append(cur.lastrowid)
        self.conn.commit()
        return ids

    def get_message_text(self, message_id: int) -> str | None:
        row = self.conn.execute("SELECT content FROM messages WHERE id = ?", (message_id,)).fetchone()
        return row[0] if row else None

    def set_message_status(self, message_id: int, status: str) -> None:
        self.conn.execute("UPDATE messages SET status = ? WHERE id = ?", (status, message_id))
        self.conn.commit()

    def chunk_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ----- index vectoriels -----------------------------------------------------

    def ensure_vec_table(self, embedder_key: str, dimension: int) -> str:
        table = _vec_table(embedder_key)
        if table not in self._vec_dims:
            self.conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dimension}])")
            self.conn.commit()
            self._vec_dims[table] = dimension
        return table

    def pending_chunks(self, embedder_key: str, dimension: int, limit: int = 256) -> list[tuple[int, str]]:
        """Chunks sans vecteur pour cet embedder (pour le ré-embedding de fond)."""
        table = self.ensure_vec_table(embedder_key, dimension)
        rows = self.conn.execute(
            f"SELECT c.id, c.content FROM chunks c WHERE c.id NOT IN (SELECT rowid FROM {table}) ORDER BY c.id LIMIT ?",
            (limit,),
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def store_vectors(
        self, embedder_key: str, dimension: int, chunk_ids: list[int], vectors: list[list[float]]
    ) -> None:
        table = self.ensure_vec_table(embedder_key, dimension)
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table}(rowid, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vector)),
            )
        self.conn.commit()

    def knn(self, embedder_key: str, dimension: int, query: list[float], k: int) -> list[tuple[int, float]]:
        """(rowid, distance L2) : les vecteurs étant normalisés, cosinus = 1 - d² / 2."""
        table = self.ensure_vec_table(embedder_key, dimension)
        rows = self.conn.execute(
            f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(query), k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def bm25(self, text: str, k: int) -> list[tuple[int, float]]:
        """(rowid, score bm25 FTS5) : plus le score est négatif, plus le résultat est pertinent."""
        match = _fts_query(text)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    # ----- recherche hybride ---------------------------------------------------

    def hybrid_search(
        self,
        embedder_key: str,
        dimension: int,
        query_text: str,
        query_vector: list[float],
        *,
        candidates: int,
        limit: int,
        rrf_k: int,
        exclude_message_ids: set[int] | None = None,
        min_similarity: float = -math.inf,
        min_bm25: float = 0.0,
        relative_score: float = 0.0,
    ) -> list[Hit]:
        """Filtres : similarité cosinus minimale (vectoriel), pertinence BM25 minimale pour les
        résultats mot-clé seuls, et score RRF minimal relatif au meilleur résultat."""
        vector_ids = [
            rid
            for rid, distance in self.knn(embedder_key, dimension, query_vector, candidates)
            if 1 - distance * distance / 2 >= min_similarity
        ]
        keyword = self.bm25(query_text, candidates)
        keyword_ids = [rid for rid, score in keyword if rid in vector_ids or -score >= min_bm25]
        scores = reciprocal_rank_fusion([vector_ids, keyword_ids], rrf_k)
        if not scores:
            return []
        best = max(scores.values())
        scores = {rid: s for rid, s in scores.items() if s >= relative_score * best}
        ids = list(scores)
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT c.id, c.message_id, c.role, c.content FROM chunks c WHERE c.id IN ({placeholders})",
            ids,
        ).fetchall()
        excluded = exclude_message_ids or set()
        hits = [
            Hit(int(r[0]), int(r[1]), str(r[2]), str(r[3]), scores[int(r[0])])
            for r in rows
            if int(r[1]) not in excluded
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fusion RRF : score(d) = somme sur les listes de 1 / (k + rang), rang à partir de 1."""
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores
