import tempfile
import unittest
from pathlib import Path

from memory_engine.db import MemoryStore
from memory_engine.metrics import calculate_goal_progress


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

    def test_missing_goal_raises(self):
        with self.assertRaises(ValueError):
            calculate_goal_progress(self.store, 999)


if __name__ == "__main__":
    unittest.main()
