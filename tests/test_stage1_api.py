import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure importing the FastAPI module never creates a database in the source tree.
_API_DB_DIR = tempfile.mkdtemp(prefix="ame-api-import-")
os.environ["MEMORY_DB"] = str(Path(_API_DB_DIR) / "api-import.db")

from fastapi.testclient import TestClient
from memory_engine import api
from memory_engine.db import MemoryStore
from memory_engine.stage1_chat import StageOneChat
from memory_engine.workout_data import WorkoutRepository


class FakeLLM:
    def chat_json(self, system_prompt, user_prompt):
        return '''{"intent":"query","query_type":"best_set","exercise_name":"pull-ups",
        "session_label":null,"session_date":null,"load_type":"bodyweight","log":{},"clarification":""}'''


class StageOneAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.store = MemoryStore(self.db_path)
        goal_id = self.store.create_goal("Gym logs", "gym_log_entries", 100, "entries")
        self.store.record_progress(goal_id, 1, "Back day", "2026-10-13", payload={
            "record_type": "gym_session_log", "source_record_key": "gym-2026-10-13-back",
            "exercises": [{"name": "Pull-ups", "sets": [{"reps": 7, "load": "bodyweight"}]}],
            "notes": []
        })
        self.repo = WorkoutRepository(self.db_path)
        self.client = TestClient(api.app)
        self.patches = [
            patch.object(api, "store", self.store),
            patch.object(api, "workout_repository", self.repo),
            patch.object(api, "get_stage1_chat", lambda: StageOneChat(self.store, self.repo, FakeLLM())),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp_dir.cleanup()

    def test_chat_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Personal Memory — Stage 1", response.text)
        self.assertIn("Confirm and save this workout", response.text)

    def test_chat_reads_history(self):
        response = self.client.post("/chat", json={"question": "What's my best bodyweight pull-up set?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("7 reps", response.json()["answer"])

    def test_confirm_log_persists_new_exercise(self):
        response = self.client.post("/chat/confirm-log", json={
            "session_date": "2026-10-19", "session_label": "back",
            "exercises": [{"name": "Seated row", "sets": [
                {"reps": 12, "load_kg": 25}, {"reps": 12, "load_kg": 25}, {"reps": 12, "load_kg": 25}
            ]}], "notes": []
        })
        self.assertEqual(response.status_code, 201, response.text)
        result = self.repo.query({"query_type": "best_set", "exercise_name": "seated rows"})
        self.assertEqual(result["best_set"]["reps"], 12)
        self.assertEqual(result["best_set"]["load_kg"], 25)


if __name__ == "__main__":
    unittest.main()
