from __future__ import annotations

from typing import Any

from .db import MemoryStore


def calculate_goal_progress(store: MemoryStore, goal_id: int) -> dict[str, Any]:
    """Calculate progress from persisted events; do not ask the LLM to do arithmetic."""
    goal = store.get_goal(goal_id)
    if goal is None:
        raise ValueError(f"Goal {goal_id} does not exist")

    events = store.get_events(goal_id, limit=100000)
    valued_events = [event for event in events if event["value"] is not None]
    if goal["aggregation"] == "sum":
        current = sum(float(event["value"]) for event in valued_events)
    else:
        current = float(valued_events[0]["value"]) if valued_events else 0.0

    target = float(goal["target_value"])
    percent = (current / target * 100.0) if target != 0 else None
    remaining = target - current
    return {
        "goal_id": goal_id,
        "title": goal["title"],
        "metric_name": goal["metric_name"],
        "current_value": current,
        "target_value": target,
        "unit": goal["unit"],
        "aggregation": goal["aggregation"],
        "percent_complete": round(percent, 2) if percent is not None else None,
        "remaining": remaining,
        "event_count": len(valued_events),
        "latest_event_at": valued_events[0]["occurred_at"] if valued_events else None,
        "recent_events": valued_events[:10],
    }
