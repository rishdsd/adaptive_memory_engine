from __future__ import annotations

from contextlib import contextmanager

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    target_value REAL NOT NULL,
    unit TEXT NOT NULL,
    aggregation TEXT NOT NULL CHECK (aggregation IN ('sum', 'latest')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL REFERENCES goals(id),
    event_type TEXT NOT NULL,
    value REAL,
    note TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    source TEXT NOT NULL DEFAULT 'user',
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_goal_occurred
ON events(goal_id, occurred_at);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_type TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    confidence TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'rejected', 'modified')),
    feedback_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


class MemoryStore:
    """SQLite persistence for goals and append-only progress events."""

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def create_goal(
        self,
        title: str,
        metric_name: str,
        target_value: float,
        unit: str,
        aggregation: str = "sum",
    ) -> int:
        if not title.strip() or not metric_name.strip() or not unit.strip():
            raise ValueError("title, metric_name, and unit must be non-empty")
        if aggregation not in {"sum", "latest"}:
            raise ValueError("aggregation must be 'sum' or 'latest'")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO goals (title, metric_name, target_value, unit, aggregation)
                   VALUES (?, ?, ?, ?, ?)""",
                (title.strip(), metric_name.strip(), float(target_value), unit.strip(), aggregation),
            )
            return int(cur.lastrowid)

    def list_goals(self, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM goals"
        if active_only:
            query += " WHERE status = 'active'"
        query += " ORDER BY id"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query).fetchall()]

    def get_goal(self, goal_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
            return dict(row) if row else None

    def record_progress(
        self,
        goal_id: int,
        value: float,
        note: str = "",
        occurred_at: str | None = None,
        source: str = "user",
        payload: dict[str, Any] | None = None,
    ) -> int:
        from datetime import datetime, timezone

        if self.get_goal(goal_id) is None:
            raise ValueError(f"Goal {goal_id} does not exist")
        timestamp = occurred_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (goal_id, event_type, value, note, occurred_at, source, payload_json)
                   VALUES (?, 'progress', ?, ?, ?, ?, ?)""",
                (goal_id, float(value), note.strip(), timestamp, source, json.dumps(payload or {})),
            )
            conn.execute(
                "UPDATE goals SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (goal_id,),
            )
            return int(cur.lastrowid)



    def create_recommendation(
        self,
        recommendation_type: str,
        title: str,
        body: str,
        evidence: dict[str, Any],
        confidence: str = "medium",
    ) -> int:
        if not recommendation_type.strip() or not title.strip() or not body.strip():
            raise ValueError("recommendation_type, title, and body must be non-empty")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO recommendations
                   (recommendation_type, title, body, evidence_json, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (recommendation_type.strip(), title.strip(), body.strip(),
                 json.dumps(evidence, ensure_ascii=False), confidence),
            )
            return int(cur.lastrowid)

    def list_recommendations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM recommendations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_recommendation(row) for row in rows]

    def get_recommendation(self, recommendation_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)
            ).fetchone()
        return self._decode_recommendation(row) if row else None

    @staticmethod
    def _decode_recommendation(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json"))
        return item

    def record_recommendation_feedback(
        self, recommendation_id: int, status: str, note: str = ""
    ) -> None:
        if status not in {"accepted", "rejected", "modified"}:
            raise ValueError("status must be accepted, rejected, or modified")
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE recommendations
                   SET status = ?, feedback_note = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                   WHERE id = ?""",
                (status, note.strip(), recommendation_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Recommendation {recommendation_id} does not exist")

    def get_events(self, goal_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM events WHERE goal_id = ?
                   ORDER BY occurred_at DESC, id DESC LIMIT ?""",
                (goal_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result
