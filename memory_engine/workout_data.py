from __future__ import annotations

from contextlib import contextmanager

import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

WORKOUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS workout_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_event_id INTEGER UNIQUE REFERENCES events(id) ON DELETE CASCADE,
    session_date TEXT NOT NULL,
    session_label TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    record_completeness TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_workout_sessions_date ON workout_sessions(session_date);
CREATE INDEX IF NOT EXISTS idx_workout_sessions_label ON workout_sessions(session_label);
CREATE TABLE IF NOT EXISTS workout_exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES workout_sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    exercise_name TEXT NOT NULL,
    exercise_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workout_exercises_key ON workout_exercises(exercise_key);
CREATE TABLE IF NOT EXISTS exercise_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workout_exercise_id INTEGER NOT NULL REFERENCES workout_exercises(id) ON DELETE CASCADE,
    set_number INTEGER NOT NULL,
    reps INTEGER,
    load TEXT,
    load_kg REAL,
    load_kg_per_hand REAL,
    assistance_kg REAL,
    machine_level REAL,
    machine_setting REAL,
    other_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_exercise_sets_reps ON exercise_sets(reps);
"""


def normalize_exercise_name(name: str) -> str:
    """Normalize superficial spelling differences without a hardcoded exercise catalogue."""
    key = re.sub(r"[^a-z0-9]", "", name.casefold())
    if len(key) > 4 and key.endswith("s"):
        key = key[:-1]
    return key


def _session_label(payload: dict[str, Any], note: str) -> str:
    source_key = str(payload.get("source_record_key", ""))
    match = re.match(r"^gym-\d{4}-\d{2}-\d{2}-(.+)$", source_key)
    if match:
        return match.group(1).replace("-", " ").strip().lower()
    text = f"{source_key} {note}".lower()
    for label in ("back biceps", "back", "chest", "shoulders", "legs", "arms"):
        if label in text:
            return label
    return ""


class WorkoutRepository:
    """Normalized read projection over the append-only events table.

    Existing event payloads remain the canonical record. The normalized tables are
    rebuildable indexes for reliable querying, not a replacement for the source log.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.initialize_and_backfill()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
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

    def initialize_and_backfill(self) -> None:
        with self._connect() as conn:
            conn.executescript(WORKOUT_SCHEMA)
            rows = conn.execute(
                "SELECT id, occurred_at, note, source, payload_json FROM events ORDER BY id"
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except json.JSONDecodeError:
                    continue
                if payload.get("record_type") == "gym_session_log":
                    self._sync_event(conn, dict(row), payload)

    def sync_event(self, event_id: int) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, occurred_at, note, source, payload_json FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Event {event_id} does not exist")
            payload = json.loads(row["payload_json"] or "{}")
            if payload.get("record_type") != "gym_session_log":
                return
            self._sync_event(conn, dict(row), payload)

    def _sync_event(self, conn: sqlite3.Connection, event: dict[str, Any], payload: dict[str, Any]) -> None:
        event_id = int(event["id"])
        exists = conn.execute(
            "SELECT id FROM workout_sessions WHERE source_event_id = ?", (event_id,)
        ).fetchone()
        if exists:
            return  # Idempotent backfill: never duplicate imported history.
        session_date = str(event.get("occurred_at") or date.today().isoformat())[:10]
        cur = conn.execute(
            """INSERT INTO workout_sessions
               (source_event_id, session_date, session_label, note, source, record_completeness)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, session_date, _session_label(payload, event.get("note", "")),
             event.get("note", ""), event.get("source", "user"),
             payload.get("record_completeness", "unknown")),
        )
        session_id = int(cur.lastrowid)
        for position, exercise in enumerate(payload.get("exercises") or [], 1):
            name = str(exercise.get("name") or "").strip()
            if not name:
                continue
            ex_cur = conn.execute(
                """INSERT INTO workout_exercises
                   (session_id, position, exercise_name, exercise_key) VALUES (?, ?, ?, ?)""",
                (session_id, position, name, normalize_exercise_name(name)),
            )
            exercise_id = int(ex_cur.lastrowid)
            for set_number, set_data in enumerate(exercise.get("sets") or [], 1):
                known = {"reps", "load", "load_kg", "load_kg_per_hand", "assistance_kg",
                         "machine_level", "machine_setting"}
                other = {k: v for k, v in set_data.items() if k not in known}
                conn.execute(
                    """INSERT INTO exercise_sets
                       (workout_exercise_id, set_number, reps, load, load_kg,
                        load_kg_per_hand, assistance_kg, machine_level, machine_setting, other_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (exercise_id, set_number, set_data.get("reps"), set_data.get("load"),
                     set_data.get("load_kg"), set_data.get("load_kg_per_hand"),
                     set_data.get("assistance_kg"), set_data.get("machine_level"),
                     set_data.get("machine_setting"), json.dumps(other, ensure_ascii=False)),
                )

    def create_workout_event(
        self,
        store: Any,
        session_date: str,
        session_label: str,
        exercises: list[dict[str, Any]],
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist the canonical event first, then index it for deterministic reads."""
        goal = next((g for g in store.list_goals() if g.get("metric_name") == "gym_log_entries"), None)
        if goal is None:
            goal_id = store.create_goal(
                "Historical Gym Log Entries", "gym_log_entries", 100, "entries", "sum"
            )
        else:
            goal_id = int(goal["id"])
        label_key = re.sub(r"[^a-z0-9]+", "-", session_label.casefold()).strip("-")
        payload = {
            "record_type": "gym_session_log",
            "source_record_key": f"gym-{session_date}-{label_key}" if label_key else f"gym-{session_date}",
            "source": "chat_log",
            "source_confidence": "user_confirmed",
            "date_precision": "date_only",
            "record_completeness": "detailed_session_log",
            "exercises": exercises,
            "notes": notes or [],
        }
        event_id = store.record_progress(
            goal_id, 1, f"{session_date} — {session_label or 'workout'} logged via chat",
            occurred_at=session_date, source="chat", payload=payload,
        )
        self.sync_event(event_id)
        return {"event_id": event_id, "goal_id": goal_id, "session_date": session_date,
                "session_label": session_label, "exercises": exercises, "notes": notes or []}

    @staticmethod
    def _set_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = {key: row[key] for key in (
            "set_number", "reps", "load", "load_kg", "load_kg_per_hand",
            "assistance_kg", "machine_level", "machine_setting")}
        item.update(json.loads(row["other_json"] or "{}"))
        return item

    def _session_records(self, where: str = "", params: tuple[Any, ...] = (), limit: int | None = 500) -> list[dict[str, Any]]:
        sql = "SELECT * FROM workout_sessions"
        if where:
            sql += " WHERE " + where
        sql += " ORDER BY session_date DESC, id DESC"
        query_params = params
        if limit is not None:
            sql += " LIMIT ?"
            query_params = (*params, limit)
        with self._connect() as conn:
            sessions = conn.execute(sql, query_params).fetchall()
            results = []
            for session in sessions:
                ex_rows = conn.execute(
                    "SELECT * FROM workout_exercises WHERE session_id = ? ORDER BY position, id",
                    (session["id"],),
                ).fetchall()
                exercises = []
                for ex in ex_rows:
                    sets = conn.execute(
                        "SELECT * FROM exercise_sets WHERE workout_exercise_id = ? ORDER BY set_number",
                        (ex["id"],),
                    ).fetchall()
                    exercises.append({"name": ex["exercise_name"], "sets": [self._set_dict(s) for s in sets]})
                results.append({"session_id": session["id"], "source_event_id": session["source_event_id"],
                    "date": session["session_date"], "session_label": session["session_label"],
                    "note": session["note"], "source": session["source"],
                    "record_completeness": session["record_completeness"], "exercises": exercises})
            return results

    @staticmethod
    def _load_category(set_data: dict[str, Any]) -> str:
        """Classify only from recorded measurements; never assume missing load means bodyweight."""
        if set_data.get("assistance_kg") is not None:
            return "assisted"
        if set_data.get("load_kg") is not None or set_data.get("load_kg_per_hand") is not None:
            return "weighted"
        if set_data.get("load") == "bodyweight":
            return "bodyweight"
        return "load_not_recorded"

    def query(self, plan: dict[str, Any]) -> dict[str, Any]:
        query_type = plan.get("query_type")
        exercise_name = str(plan.get("exercise_name") or "").strip()
        exercise_key = normalize_exercise_name(exercise_name) if exercise_name else ""
        session_label = str(plan.get("session_label") or "").strip().casefold()
        session_date = plan.get("session_date")
        load_type = plan.get("load_type")
        sessions = self._session_records(limit=100 if query_type == "list_sessions" else None)
        if session_date:
            sessions = [s for s in sessions if s["date"] == session_date]
        if session_label:
            sessions = [s for s in sessions if session_label in s["session_label"].casefold()
                        or session_label in s["note"].casefold()]

        if query_type == "list_exercises" or query_type == "session_exercises":
            names = []
            seen = set()
            for session in sessions:
                for exercise in session["exercises"]:
                    key = normalize_exercise_name(exercise["name"])
                    if key not in seen:
                        seen.add(key)
                        names.append(exercise["name"])
            return {"query_type": query_type, "filters": {"session_label": session_label or None,
                    "session_date": session_date}, "exercises": names, "sessions_considered": len(sessions)}

        matches = []
        for session in sessions:
            for exercise in session["exercises"]:
                if exercise_key and normalize_exercise_name(exercise["name"]) != exercise_key:
                    continue
                for set_data in exercise["sets"]:
                    if load_type == "bodyweight" and self._load_category(set_data) != "bodyweight":
                        continue
                    if load_type == "assisted" and self._load_category(set_data) != "assisted":
                        continue
                    if load_type == "weighted" and self._load_category(set_data) != "weighted":
                        continue
                    matches.append({"date": session["date"], "session_id": session["session_id"],
                        "session_label": session["session_label"], "exercise_name": exercise["name"], **set_data})

        if query_type == "best_set":
            with_reps = [m for m in matches if isinstance(m.get("reps"), int)]
            if plan.get("group_by_load") and load_type is None:
                grouped: dict[str, dict[str, Any]] = {}
                for item in with_reps:
                    category = self._load_category(item)
                    previous = grouped.get(category)
                    if previous is None or (item["reps"], item["date"]) > (previous["reps"], previous["date"]):
                        grouped[category] = item
                return {"query_type": query_type, "exercise_name": exercise_name,
                        "load_type": load_type, "best_sets_by_load_type": grouped,
                        "matching_set_count": len(matches)}
            result = max(with_reps, key=lambda m: (m["reps"], m["date"]), default=None)
            return {"query_type": query_type, "exercise_name": exercise_name,
                    "load_type": load_type, "best_set": result, "matching_set_count": len(matches)}
        if query_type == "latest_set":
            if not matches:
                return {"query_type": query_type, "exercise_name": exercise_name, "sets": []}
            latest_session = max(matches, key=lambda m: (m["date"], m["session_id"]))
            latest_session_id = latest_session["session_id"]
            latest = [m for m in matches if m["session_id"] == latest_session_id]
            return {"query_type": query_type, "exercise_name": exercise_name,
                    "latest_date": latest_session["date"], "sets": sorted(latest, key=lambda m: m.get("set_number", 0))}
        if query_type == "exercise_history":
            return {"query_type": query_type, "exercise_name": exercise_name,
                    "records": sorted(matches, key=lambda m: (m["date"], m.get("set_number", 0)), reverse=True)}
        if query_type == "list_sessions":
            return {"query_type": query_type, "sessions": sessions}
        return {"query_type": "unsupported", "reason": "The request could not be mapped to a supported history query."}
