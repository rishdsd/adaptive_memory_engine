import tempfile
import unittest
from pathlib import Path

from memory_engine.db import MemoryStore
from memory_engine.metrics import calculate_goal_progress
from memory_engine.workout import render_gym_history, compare_pullup_records


class MemoryEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "test.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sum_progress_is_calculated_from_events(self):
        goal_id = self.store.create_goal("Study DSA", "study_hours", 12, "hours", "sum")
        self.store.record_progress(goal_id, 1.5, "Arrays practice", "2026-10-15T08:00:00+00:00")
        self.store.record_progress(goal_id, 2.0, "Sliding window", "2026-10-16T08:00:00+00:00")

        result = calculate_goal_progress(self.store, goal_id)
        self.assertEqual(result["current_value"], 3.5)
        self.assertEqual(result["percent_complete"], 29.17)
        self.assertEqual(result["event_count"], 2)

    def test_latest_aggregation_uses_latest_event(self):
        goal_id = self.store.create_goal("Reach target weight", "weight", 75, "kg", "latest")
        self.store.record_progress(goal_id, 70.2, "Morning measurement", "2026-10-15T07:00:00+00:00")
        self.store.record_progress(goal_id, 70.8, "Morning measurement", "2026-10-16T07:00:00+00:00")

        result = calculate_goal_progress(self.store, goal_id)
        self.assertEqual(result["current_value"], 70.8)
        self.assertEqual(result["event_count"], 2)



    def test_workout_renderer_distinguishes_sets_from_reps(self):
        goal_id = self.store.create_goal(
            "Historical Gym Log Entries", "gym_log_entries", 100, "entries", "sum"
        )
        self.store.record_progress(
            goal_id, 1, "Pull-up session", "2026-09-08",
            source="historical_chat",
            payload={
                "record_type": "gym_session_log",
                "record_completeness": "partial_session_log",
                "exercises": [{
                    "name": "Pull-ups",
                    "sets": [
                        {"reps": 7, "load": "bodyweight"},
                        {"reps": 6, "load": "bodyweight"},
                        {"reps": 6, "assistance_kg": 12},
                    ],
                }],
                "notes": [],
            },
        )
        rendered, evidence = render_gym_history(self.store, goal_id)
        self.assertIn("Pull-ups: 3 recorded set(s)", rendered)
        self.assertIn("Set 1: 7 reps, bodyweight", rendered)
        self.assertIn("Set 2: 6 reps, bodyweight", rendered)
        self.assertIn("Set 3: 6 reps, 12 kg assistance", rendered)
        self.assertNotIn("7 sets", rendered)
        self.assertEqual(len(evidence["sessions"][0]["exercises"][0]["sets"]), 3)



    def test_workout_renderer_keeps_machine_settings_separate_from_load(self):
        goal_id = self.store.create_goal(
            "Historical Gym Log Entries", "gym_log_entries", 100, "entries", "sum"
        )
        self.store.record_progress(
            goal_id, 1, "Pec fly partial log", "2026-09-22",
            source="historical_chat",
            payload={
                "record_type": "gym_session_log",
                "record_completeness": "partial_session_log",
                "exercises": [{
                    "name": "Pec fly",
                    "sets": [{"machine_setting": 5, "reps": None, "load_kg": None}],
                }],
                "notes": [],
            },
        )
        rendered, _ = render_gym_history(self.store, goal_id)
        self.assertIn("Set 1: reps not recorded, machine setting 5, load not recorded", rendered)
        self.assertNotIn("5 kg", rendered)

    def test_pullup_comparison_is_cautious_and_uses_recorded_values(self):
        goal_id = self.store.create_goal(
            "Historical Gym Log Entries", "gym_log_entries", 100, "entries", "sum"
        )
        records = [
            ("2026-09-08", [
                {"reps": 7, "load": "bodyweight"},
                {"reps": 6, "load": "bodyweight"},
                {"reps": 6, "assistance_kg": 12},
            ]),
            ("2026-09-23", [
                {"reps": 7, "load": "bodyweight"},
                {"reps": 5, "load": "bodyweight"},
                {"reps": 9, "assistance_kg": 20},
            ]),
        ]
        for date, sets in records:
            self.store.record_progress(
                goal_id, 1, "Pull-up log", date, source="historical_chat",
                payload={"record_type": "gym_session_log",
                         "record_completeness": "partial_session_log",
                         "exercises": [{"name": "Pull-ups", "sets": sets}], "notes": []},
            )
        _, evidence = render_gym_history(self.store, goal_id)
        comparison = compare_pullup_records(evidence)
        self.assertEqual(comparison["2026-09-08"][0]["reps"], 7)
        self.assertEqual(comparison["2026-09-23"][1]["reps"], 5)
        self.assertEqual(comparison["2026-09-08"][2]["assistance_kg"], 12)
        self.assertEqual(comparison["2026-09-23"][2]["assistance_kg"], 20)
        self.assertTrue(any("not enough" in x.lower() or "insufficient" in x.lower()
                            for x in comparison["observations"]))

    def test_recommendation_feedback_is_persisted(self):
        recommendation_id = self.store.create_recommendation(
            "workout_next_step", "Keep conditions consistent",
            "Record each set and assistance.", {"sessions": ["2026-09-08", "2026-09-23"]},
            confidence="low",
        )
        self.store.record_recommendation_feedback(
            recommendation_id, "accepted", "Useful for my next workout"
        )
        recommendation = self.store.get_recommendation(recommendation_id)
        self.assertEqual(recommendation["status"], "accepted")
        self.assertEqual(recommendation["feedback_note"], "Useful for my next workout")
        self.assertEqual(recommendation["evidence"]["sessions"], ["2026-09-08", "2026-09-23"])

    def test_missing_goal_raises(self):
        with self.assertRaises(ValueError):
            calculate_goal_progress(self.store, 999)


if __name__ == "__main__":
    unittest.main()
