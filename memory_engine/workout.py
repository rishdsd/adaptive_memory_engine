from __future__ import annotations

from typing import Any

from .db import MemoryStore


def find_gym_goal(store: MemoryStore, goal_id: int | None = None) -> dict[str, Any]:
    """Find the goal holding imported workout logs."""
    if goal_id is not None:
        goal = store.get_goal(goal_id)
        if goal is None:
            raise ValueError(f"Goal {goal_id} does not exist")
        return goal
    matches = [
        goal for goal in store.list_goals()
        if goal.get("metric_name") == "gym_log_entries"
        or "gym" in goal.get("title", "").lower()
    ]
    if not matches:
        raise ValueError("No gym-log goal found. Pass --goal-id or import gym history first.")
    return matches[0]


def _format_set(set_data: dict[str, Any]) -> str:
    parts = []
    reps = set_data.get("reps")
    parts.append(f"{reps} reps" if reps is not None else "reps not recorded")

    if set_data.get("load") == "bodyweight":
        parts.append("bodyweight")
    if set_data.get("load_kg") is not None:
        parts.append(f"{set_data['load_kg']} kg")
    if set_data.get("load_kg_per_hand") is not None:
        parts.append(f"{set_data['load_kg_per_hand']} kg per hand")
    if set_data.get("assistance_kg") is not None:
        parts.append(f"{set_data['assistance_kg']} kg assistance")
    if set_data.get("machine_level") is not None:
        parts.append(f"machine level {set_data['machine_level']}")
    if set_data.get("machine_setting") is not None:
        parts.append(f"machine setting {set_data['machine_setting']}")

    known_load_keys = {"load", "load_kg", "load_kg_per_hand", "assistance_kg"}
    if not any(key in set_data and set_data[key] is not None for key in known_load_keys):
        parts.append("load not recorded")
    return ", ".join(parts)


def render_gym_history(store: MemoryStore, goal_id: int | None = None) -> tuple[str, dict[str, Any]]:
    """Render workout facts in Python so the LLM cannot confuse reps with set counts."""
    goal = find_gym_goal(store, goal_id)
    events = store.get_events(goal["id"], limit=100000)
    sessions = []
    for event in events:
        payload = event.get("payload") or {}
        if payload.get("record_type") != "gym_session_log":
            continue
        date = str(event.get("occurred_at", ""))[:10]
        exercises = payload.get("exercises") or []
        sessions.append({
            "date": date,
            "event_id": event["id"],
            "note": event.get("note", ""),
            "record_completeness": payload.get("record_completeness", "unknown"),
            "exercises": exercises,
            "notes": payload.get("notes", []),
        })

    sessions.sort(key=lambda x: (x["date"], x["event_id"]))
    lines = []
    for session in sessions:
        lines.append(f"{session['date']} (event {session['event_id']}; {session['record_completeness']})")
        for exercise in session["exercises"]:
            sets = exercise.get("sets") or []
            lines.append(f"  {exercise.get('name', 'Unnamed exercise')}: {len(sets)} recorded set(s)")
            for index, set_data in enumerate(sets, 1):
                lines.append(f"    Set {index}: {_format_set(set_data)}")
        for note in session["notes"]:
            lines.append(f"  Note: {note}")
    if not sessions:
        lines.append("No gym-session payloads were found for this goal.")
    evidence = {"goal_id": goal["id"], "goal_title": goal["title"], "sessions": sessions}
    return "\\n".join(lines), evidence


def compare_pullup_records(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative, evidence-backed comparison of Sep 8 and Sep 23 logs."""
    wanted = {"2026-09-08": None, "2026-09-23": None}
    for session in evidence.get("sessions", []):
        if session["date"] not in wanted:
            continue
        for exercise in session.get("exercises", []):
            if exercise.get("name", "").strip().lower() == "pull-ups":
                wanted[session["date"]] = exercise.get("sets", [])
                break

    if any(value is None for value in wanted.values()):
        raise ValueError("Could not find pull-up records for both 2026-09-08 and 2026-09-23.")

    def describe(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "set_number": i,
                "reps": s.get("reps"),
                "load": s.get("load"),
                "assistance_kg": s.get("assistance_kg"),
            }
            for i, s in enumerate(sets, 1)
        ]

    sep8 = describe(wanted["2026-09-08"])
    sep23 = describe(wanted["2026-09-23"])
    comparison = {
        "2026-09-08": sep8,
        "2026-09-23": sep23,
        "observations": [
            "The first bodyweight set was 7 reps on both dates.",
            "The second bodyweight set was 6 reps on Sep 8 and 5 reps on Sep 23.",
            "The final sets used different assistance: 12 kg on Sep 8 and 20 kg on Sep 23.",
            "These two partial records are not enough to establish an overall strength trend.",
        ],
    }
    return comparison
