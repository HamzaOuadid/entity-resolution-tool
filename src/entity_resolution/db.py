"""SQLite persistence for the four data-model tables.

SQLite is used instead of Postgres (the spec's suggestion) because this
environment has no Postgres/Docker available, and the pipeline has no need
for anything Postgres-specific -- it's a batch job over a modest number of
rows. See README "Risks / Open Questions" for the full rationale.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .models import CandidatePair, NormalizedEntity, RawEntity, ResolutionDecision

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    raw_name TEXT NOT NULL,
    raw_address TEXT NOT NULL,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS normalized_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_entity_id INTEGER NOT NULL REFERENCES raw_entities(id),
    normalized_name TEXT NOT NULL,
    normalized_address TEXT NOT NULL,
    normalized_city TEXT NOT NULL DEFAULT '',
    normalized_postal_code TEXT NOT NULL DEFAULT '',
    legal_suffix TEXT NOT NULL DEFAULT '',
    name_core_tokens TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    UNIQUE(raw_entity_id)
);

CREATE TABLE IF NOT EXISTS candidate_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_a_id INTEGER NOT NULL REFERENCES normalized_entities(id),
    entity_b_id INTEGER NOT NULL REFERENCES normalized_entities(id),
    blocking_key TEXT NOT NULL,
    similarity_score REAL,
    UNIQUE(entity_a_id, entity_b_id)
);

CREATE TABLE IF NOT EXISTS resolution_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL REFERENCES candidate_pairs(id),
    decision TEXT NOT NULL,
    reviewer TEXT,
    hand_label TEXT,
    UNIQUE(pair_id)
);

CREATE INDEX IF NOT EXISTS idx_candidate_pairs_a ON candidate_pairs(entity_a_id);
CREATE INDEX IF NOT EXISTS idx_candidate_pairs_b ON candidate_pairs(entity_b_id);
"""


class EntityResolutionDB:
    """Thin wrapper around a SQLite file matching the spec's data model."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EntityResolutionDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        for table in (
            "resolution_decisions",
            "candidate_pairs",
            "normalized_entities",
            "raw_entities",
        ):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    # -- raw_entities ---------------------------------------------------
    def insert_raw_entity(self, entity: RawEntity) -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO raw_entities (source, external_id, raw_name, raw_address) "
            "VALUES (?, ?, ?, ?)",
            (entity.source, entity.external_id, entity.raw_name, entity.raw_address),
        )
        self.conn.commit()
        if cur.lastrowid and cur.rowcount:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM raw_entities WHERE source=? AND external_id=?",
            (entity.source, entity.external_id),
        ).fetchone()
        return row["id"]

    def get_raw_entities(self) -> list[RawEntity]:
        rows = self.conn.execute("SELECT * FROM raw_entities ORDER BY id").fetchall()
        return [
            RawEntity(
                id=r["id"],
                source=r["source"],
                external_id=r["external_id"],
                raw_name=r["raw_name"],
                raw_address=r["raw_address"],
            )
            for r in rows
        ]

    # -- normalized_entities ---------------------------------------------
    def insert_normalized_entity(self, entity: NormalizedEntity) -> int:
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO normalized_entities "
            "(raw_entity_id, normalized_name, normalized_address, normalized_city, "
            "normalized_postal_code, legal_suffix, name_core_tokens, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entity.raw_entity_id,
                entity.normalized_name,
                entity.normalized_address,
                entity.normalized_city,
                entity.normalized_postal_code,
                entity.legal_suffix,
                ",".join(entity.name_core_tokens),
                entity.source,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_normalized_entities(self) -> list[NormalizedEntity]:
        rows = self.conn.execute("SELECT * FROM normalized_entities ORDER BY id").fetchall()
        return [_row_to_normalized(r) for r in rows]

    # -- candidate_pairs ---------------------------------------------------
    def insert_candidate_pair(self, pair: CandidatePair) -> int:
        a, b = sorted((pair.entity_a_id, pair.entity_b_id))
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO candidate_pairs (entity_a_id, entity_b_id, blocking_key, similarity_score) "
            "VALUES (?, ?, ?, ?)",
            (a, b, pair.blocking_key, pair.similarity_score),
        )
        self.conn.commit()
        if cur.lastrowid and cur.rowcount:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM candidate_pairs WHERE entity_a_id=? AND entity_b_id=?", (a, b)
        ).fetchone()
        return row["id"]

    def update_pair_score(self, pair_id: int, score: float) -> None:
        self.conn.execute(
            "UPDATE candidate_pairs SET similarity_score=? WHERE id=?", (score, pair_id)
        )
        self.conn.commit()

    def get_candidate_pairs(self) -> list[CandidatePair]:
        rows = self.conn.execute("SELECT * FROM candidate_pairs ORDER BY id").fetchall()
        return [
            CandidatePair(
                id=r["id"],
                entity_a_id=r["entity_a_id"],
                entity_b_id=r["entity_b_id"],
                blocking_key=r["blocking_key"],
                similarity_score=r["similarity_score"],
            )
            for r in rows
        ]

    # -- resolution_decisions ----------------------------------------------
    def upsert_decision(self, decision: ResolutionDecision) -> int:
        cur = self.conn.execute(
            "INSERT INTO resolution_decisions (pair_id, decision, reviewer, hand_label) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(pair_id) DO UPDATE SET decision=excluded.decision, "
            "reviewer=COALESCE(excluded.reviewer, resolution_decisions.reviewer), "
            "hand_label=COALESCE(excluded.hand_label, resolution_decisions.hand_label)",
            (decision.pair_id, decision.decision, decision.reviewer, decision.hand_label),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM resolution_decisions WHERE pair_id=?", (decision.pair_id,)
        ).fetchone()
        return row["id"]

    def get_decisions(self) -> list[ResolutionDecision]:
        rows = self.conn.execute("SELECT * FROM resolution_decisions ORDER BY id").fetchall()
        return [
            ResolutionDecision(
                id=r["id"],
                pair_id=r["pair_id"],
                decision=r["decision"],
                reviewer=r["reviewer"],
                hand_label=r["hand_label"],
            )
            for r in rows
        ]


def _row_to_normalized(r: sqlite3.Row) -> NormalizedEntity:
    tokens = tuple(t for t in r["name_core_tokens"].split(",") if t)
    return NormalizedEntity(
        id=r["id"],
        raw_entity_id=r["raw_entity_id"],
        normalized_name=r["normalized_name"],
        normalized_address=r["normalized_address"],
        normalized_city=r["normalized_city"],
        normalized_postal_code=r["normalized_postal_code"],
        legal_suffix=r["legal_suffix"],
        name_core_tokens=tokens,
        source=r["source"],
    )
