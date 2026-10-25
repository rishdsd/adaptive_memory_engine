import json

from memory_engine.stage1_chat import StageOneChat
from memory_engine.workout_data import WorkoutRepository


class FakeStore:
    pass


class FakeLLM:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def chat_json(self, system, user):
        self.calls += 1
        return json.dumps(self.plan)


class FakeRepository:
    def __init__(self, result):
        self.result = result
        self.last_plan = None

    def query(self, plan):
        self.last_plan = dict(plan)
        if plan.get("query_type") == "list_exercises":
            return {"exercises": ["Pull-ups", "Push-ups"]}
        return self.result


def make_plan(**overrides):
    plan = {
        "intent": "query", "query_type": "exercise_history", "exercise_name": "pull up",
        "session_label": None, "session_date": None, "load_type": None,
        "include_dates": False, "log": {}, "clarification": ""
    }
    plan.update(overrides)
    return plan


def test_history_without_dates_is_a_presentation_option():
    llm = FakeLLM(make_plan())
    repo = FakeRepository({"query_type": "exercise_history", "exercise_name": "Pull-ups",
                           "records": [{"date": "2026-10-13", "set_number": 1, "reps": 7, "load": "bodyweight"}]})
    chat = StageOneChat(FakeStore(), repo, llm)
    response = chat.process("Show me my pull up story with out date")
    assert response["mode"] == "history_answer"
    assert "2026-10-13" not in response["answer"]
    assert "7 reps" in response["answer"]
    assert llm.calls == 1  # No phrase-specific regex routing intercepted the message.


def test_best_set_without_load_filter_groups_categories():
    llm = FakeLLM(make_plan(query_type="best_set", include_dates=True))
    repo = FakeRepository({"query_type": "best_set", "exercise_name": "Pull-ups",
                           "best_sets_by_load_type": {
                               "bodyweight": {"date": "2026-10-13", "reps": 7, "load": "bodyweight", "exercise_name": "Pull-ups"},
                               "assisted": {"date": "2026-09-23", "reps": 9, "assistance_kg": 20, "exercise_name": "Pull-ups"}
                           }})
    response = StageOneChat(FakeStore(), repo, llm).process("What is my best pull-up set?")
    assert repo.last_plan["group_by_load"] is True
    assert "bodyweight" in response["answer"]
    assert "assisted" in response["answer"]


def test_load_categories_do_not_assume_missing_load_is_bodyweight():
    category = WorkoutRepository._load_category
    assert category({"reps": 7, "load": "bodyweight"}) == "bodyweight"
    assert category({"reps": 9, "assistance_kg": 20}) == "assisted"
    assert category({"reps": 5, "load_kg": 5}) == "weighted"
    assert category({"reps": 7}) == "load_not_recorded"
