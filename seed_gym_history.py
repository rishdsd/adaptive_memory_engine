"""
Seed source-grounded historical gym logs into the Adaptive Memory Engine.

Run from the adaptive_memory_engine project root:
    python seed_gym_history.py

Safe to rerun: source_record_key in each event payload prevents duplicate imports.
Only workout details explicitly present in prior conversation records are included.
"""

from __future__ import annotations

import json
from memory_engine.db import MemoryStore

GOAL_TITLE = "Historical Gym Log Entries"

# Dates are known, but exact workout times were not consistently recorded.
# We store date-only strings and mark date_precision="date_only" in each payload.
RECORDS = [
    {
        "key": "gym-2026-09-08-back-biceps",
        "date": "2026-09-08",
        "note": "Back + Biceps; explicitly logged exercise-level session",
        "completeness": "detailed_session_log",
        "exercises": [
            {
                "name": "Pull-ups",
                "sets": [
                    {"reps": 7, "load": "bodyweight"},
                    {"reps": 6, "load": "bodyweight"},
                    {"reps": 6, "assistance_kg": 12},
                ],
            },
            {
                "name": "Lat pulldown",
                "sets": [
                    {"reps": 12, "machine_level": 6},
                    {"reps": 10, "machine_level": 7},
                    {"reps": 8, "machine_level": 8},
                ],
            },
            {
                "name": "Reverse pec deck",
                "sets": [
                    {"reps": 10, "machine_level": 4},
                    {"reps": 10, "machine_level": 4},
                    {"reps": 8, "machine_level": 5},
                ],
            },
            {
                "name": "One-arm dumbbell row",
                "sets": [
                    {"reps": 12, "load_kg": 10},
                    {"reps": 10, "load_kg": 12.5},
                    {"reps": 10, "load_kg": 15},
                ],
            },
            {
                "name": "Biceps curls",
                "sets": [
                    {"reps": 12, "load_kg": 5},
                    {"reps": 10, "load_kg": 7.5},
                    {"reps": 5, "load_kg": 10},
                ],
            },
            {
                "name": "Hammer curls",
                "sets": [
                    {"reps": 12, "load_kg": 5},
                    {"reps": 7, "load_kg": 7.5},
                    {"reps": 5, "load_kg": 5},
                ],
            },
        ],
        "notes": [
            "No workout duration or additional notes were recorded.",
            "Machine levels are retained as levels/settings; they are not assumed to be kilograms.",
        ],
    },
    {
        "key": "gym-2026-09-11-shoulders",
        "date": "2026-09-11",
        "note": "Shoulder exercise log",
        "completeness": "partial_session_log",
        "exercises": [
            {
                "name": "Shoulder press",
                "sets": [
                    {"reps": 12, "load_kg_per_hand": 7.5},
                    {"reps": 12, "load_kg_per_hand": 10},
                    {"reps": 9, "load_kg_per_hand": 12.5},
                ],
            },
            {
                "name": "Lateral raises",
                "sets": [
                    {"reps": 15, "load_kg": 5},
                    {"reps": 15, "load_kg": 5},
                    {"reps": 15, "load_kg": 5},
                ],
            },
        ],
        "notes": [
            "Only these shoulder exercises are present in the retrieved record; do not infer the rest of the session."
        ],
    },
    {
        "key": "gym-2026-09-22-chest",
        "date": "2026-09-22",
        "note": "Chest session log; triceps performance not recorded",
        "completeness": "partial_session_log",
        "exercises": [
            {
                "name": "Push-ups",
                "sets": [{"reps": 15}],
            },
            {
                "name": "Dumbbell bench press",
                "sets": [
                    {"reps": 15, "load_kg_per_hand": 10},
                    {"reps": 12, "load_kg_per_hand": 12.5},
                    {"reps": 12, "load_kg_per_hand": 15},
                ],
            },
            {
                "name": "Incline dumbbell press",
                "sets": [
                    {"reps": 15, "load_kg_per_hand": 10},
                    {"reps": 12, "load_kg_per_hand": 12.5},
                    {"reps": 15, "load_kg_per_hand": 15},
                ],
            },
            {
                "name": "Pec fly",
                "sets": [
                    {"machine_setting": 5, "reps": None, "load_kg": None},
                    {"machine_setting": 6, "reps": None, "load_kg": None},
                    {"machine_setting": 7, "reps": None, "load_kg": None},
                ],
            },
        ],
        "notes": [
            "The pec-fly values 5, 6, 7 were recorded as machine inputs/settings; reps and kg were not recorded.",
            "A chest + triceps session was discussed, but no triceps exercise performance was logged.",
        ],
    },
    {
        "key": "gym-2026-09-23-back-pullups",
        "date": "2026-09-23",
        "note": "Back-session pull-up log only; other exercise completion unconfirmed",
        "completeness": "partial_session_log",
        "exercises": [
            {
                "name": "Pull-ups",
                "sets": [
                    {"reps": 7, "load": "bodyweight"},
                    {"reps": 5, "load": "bodyweight"},
                    {"reps": 9, "assistance_kg": 20},
                ],
            },
        ],
        "notes": [
            "The source does not establish additional completed exercises for this date."
        ],
    },
    {
        "key": "gym-2026-10-13-back-pullups",
        "date": "2026-10-13",
        "note": "Back-session pull-up log only; 30-minute cap and 1-hour badminton noted",
        "completeness": "partial_session_log",
        "exercises": [
            {
                "name": "Pull-ups",
                "sets": [{"reps": 7, "load": "bodyweight"}],
            },
        ],
        "notes": [
            "A 30-minute workout cap and 1-hour badminton session were mentioned.",
            "Other exercises were planned in the surrounding discussion but are not counted as completed.",
        ],
    },
]


def main() -> None:
    store = MemoryStore()
    existing_goals = store.list_goals()
    goal = next((g for g in existing_goals if g["title"] == GOAL_TITLE), None)

    if goal:
        goal_id = int(goal["id"])
        print(f"Using existing goal #{goal_id}: {GOAL_TITLE}")
    else:
        goal_id = store.create_goal(
            title=GOAL_TITLE,
            metric_name="gym_log_entries",
            target_value=100,
            unit="entries",
            aggregation="sum",
        )
        print(f"Created goal #{goal_id}: {GOAL_TITLE}")

    existing_events = store.get_events(goal_id, limit=100000)
    existing_keys = {
        event.get("payload", {}).get("source_record_key")
        for event in existing_events
    }

    imported = 0
    skipped = 0

    for record in RECORDS:
        if record["key"] in existing_keys:
            print(f"SKIP (already imported): {record['date']} — {record['note']}")
            skipped += 1
            continue

        payload = {
            "record_type": "gym_session_log",
            "source_record_key": record["key"],
            "source": "prior_chat_history",
            "source_confidence": "explicitly_logged_details",
            "date_precision": "date_only",
            "record_completeness": record["completeness"],
            "exercises": record["exercises"],
            "notes": record["notes"],
        }
        store.record_progress(
            goal_id=goal_id,
            value=1,
            note=f"{record['date']} — {record['note']}",
            occurred_at=record["date"],
            source="historical_chat",
            payload=payload,
        )
        print(f"IMPORTED: {record['date']} — {record['note']}")
        imported += 1

    print("\nImport summary")
    print("--------------")
    print(f"Goal ID: {goal_id}")
    print(f"New records imported: {imported}")
    print(f"Existing records skipped: {skipped}")
    print(f"Total source-grounded gym log entries: {len(RECORDS)}")
    print("No leg-day workout was imported: no confirmed dated exercise-level leg log was found.")
    print("No Oct 12 chest/triceps workout was imported: that date was not supported by the retrieved records.")
    print("\nProgress summary:")
    from memory_engine.metrics import calculate_goal_progress
    print(json.dumps(calculate_goal_progress(store, goal_id), indent=2))


if __name__ == "__main__":
    main()
