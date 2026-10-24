import json
import tempfile
import unittest
from pathlib import Path

from memory_engine.db import MemoryStore
from memory_engine.stage1_chat import StageOneChat
from memory_engine.workout_data import WorkoutRepository


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat_json(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        return json.dumps(self.responses.pop(0))


class StageOneChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.store = MemoryStore(self.db_path)
        self.gym_goal_id = self.store.create_goal(
            "Historical Gym Log Entries", "gym_log_entries", 100, "entries", "sum"
        )
        self.store.record_progress(
            self.gym_goal_id, 1, "Back + Biceps session", "2026-09-08", source="historical_chat",
            payload={
                "record_type": "gym_session_log",
                "source_record_key": "gym-2026-09-08-back-biceps",
                "record_completeness": "detailed_session_log",
                "exercises": [
                    {"name": "Pull-ups", "sets": [
                        {"reps": 7, "load": "bodyweight"},
                        {"reps": 6, "load": "bodyweight"},
                        {"reps": 6, "assistance_kg": 12},
                    ]},
                    {"name": "Lat pulldown", "sets": [
                        {"reps": 12, "machine_level": 6},
                        {"reps": 10, "machine_level": 7},
                    ]},
                    {"name": "One-arm dumbbell row", "sets": [{"reps": 12, "load_kg": 10}]},
                ], "notes": [],
            },
        )
        self.repo = WorkoutRepository(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backfill_is_idempotent_and_best_set_is_deterministic(self):
        # A second repository startup should not duplicate imported event rows.
        repo = WorkoutRepository(self.db_path)
        result = repo.query({"query_type": "best_set", "exercise_name": "pull up", "load_type": "bodyweight"})
        self.assertEqual(result["best_set"]["reps"], 7)
        self.assertEqual(result["best_set"]["date"], "2026-09-08")
        self.assertEqual(result["matching_set_count"], 2)
        with repo._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM workout_sessions").fetchone()[0], 1)

    def test_lists_all_exercises_for_back_day_from_records(self):
        result = self.repo.query({"query_type": "session_exercises", "session_label": "back"})
        self.assertEqual(result["exercises"], ["Pull-ups", "Lat pulldown", "One-arm dumbbell row"])

    def test_chat_returns_deterministic_answer_from_query_result(self):
        llm = FakeLLM([{
            "intent": "query", "query_type": "best_set", "exercise_name": "pull-ups",
            "session_label": None, "session_date": None, "load_type": "bodyweight",
            "log": {}, "clarification": ""
        }])
        response = StageOneChat(self.store, self.repo, llm).process("What's my best bodyweight pull-up set?")
        self.assertEqual(response["mode"], "history_answer")
        self.assertIn("7 reps", response["answer"])
        self.assertIn("2026-09-08", response["answer"])

    def test_log_requires_confirmation_before_write_then_read_after_write(self):
        llm = FakeLLM([{
            "intent": "log_workout", "query_type": None, "exercise_name": None,
            "session_label": None, "session_date": None, "load_type": None,
            "log": {"session_date": "2026-10-19", "session_label": "back",
                    "exercises": [{"name": "Seated row", "sets": [
                        {"reps": 12, "load_kg": 25}, {"reps": 12, "load_kg": 25},
                        {"reps": 12, "load_kg": 25}
                    ]}], "notes": []}, "clarification": ""
        }])
        chat = StageOneChat(self.store, self.repo, llm)
        preview = chat.process("I did seated rows today, 3 sets of 12 at 25 kg")
        self.assertEqual(preview["mode"], "log_confirmation_required")
        with self.repo._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM workout_sessions").fetchone()[0], 1)

        pending = preview["pending_log"]
        saved = self.repo.create_workout_event(
            self.store, pending["session_date"], pending["session_label"],
            pending["exercises"], pending["notes"]
        )
        self.assertEqual(saved["status"] if "status" in saved else saved["session_date"], "2026-10-19")
        result = self.repo.query({"query_type": "best_set", "exercise_name": "seated rows"})
        self.assertEqual(result["best_set"]["reps"], 12)
        self.assertEqual(result["best_set"]["load_kg"], 25)

    def test_missing_exercise_name_asks_clarification_instead_of_guessing(self):
        llm = FakeLLM([{
            "intent": "query", "query_type": "best_set", "exercise_name": None,
            "session_label": None, "session_date": None, "load_type": None,
            "log": {}, "clarification": ""
        }])
        response = StageOneChat(self.store, self.repo, llm).process("What's my best set?")
        self.assertEqual(response["mode"], "clarification")
        self.assertIn("Which exercise", response["answer"])


if __name__ == "__main__":
    unittest.main()
