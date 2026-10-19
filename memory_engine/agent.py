from __future__ import annotations

import json
from typing import Any

from .db import MemoryStore
from .metrics import calculate_goal_progress
from .ollama_client import OllamaClient
from .workout import compare_pullup_records, render_gym_history

SYSTEM_PROMPT = """You are an assistant backed by a local, persistent memory engine.
Answer using only the supplied memory context for claims about the user's goals and history.
The context includes application-calculated metrics; do not recalculate or contradict them.
Never invent missing records, dates, causes, or user preferences. If the context is insufficient,
say what is missing and ask one focused question. Distinguish recorded facts from interpretations.
When giving a recommendation, explain which evidence supports it and label uncertain hypotheses.
Be concise and practical."""


class MemoryAgent:
    def __init__(self, store: MemoryStore, llm: OllamaClient) -> None:
        self.store = store
        self.llm = llm

    def build_context(self) -> dict[str, Any]:
        goals = []
        for goal in self.store.list_goals(active_only=True):
            goals.append(calculate_goal_progress(self.store, goal["id"]))
        return {"active_goals": goals}



    def recommend_pullup_next_step(self, goal_id: int | None = None) -> dict[str, Any]:
        """Generate and persist a cautious next-step recommendation from two logged sessions."""
        _, history_evidence = render_gym_history(self.store, goal_id)
        comparison = compare_pullup_records(history_evidence)
        system_prompt = (
            "You are a cautious workout-log coach. Use only the supplied comparison. "
            "Do not change or reinterpret any numbers. Do not claim strength improved or declined: "
            "the evidence explicitly says the records are insufficient for an overall trend. "
            "Give one practical, conservative next-step recommendation and explain its evidence. "
            "Do not prescribe training to failure or assume the user has no pain/injury. "
            "Keep the response concise. Start with a short title on the first line."
        )
        user_prompt = (
            "Create one next-step recommendation based on this application-validated comparison.\n"
            + json.dumps(comparison, ensure_ascii=False, indent=2)
            + "\nRecommend a simple way to make the next pull-up log more comparable "
              "(for example, keep assistance and rest conditions consistent and record every set)."
        )
        body = self.llm.chat(system_prompt, user_prompt)
        first_line = next((line.strip().lstrip("#*- ") for line in body.splitlines() if line.strip()),
                          "Make the next pull-up session easier to compare")
        rec_id = self.store.create_recommendation(
            recommendation_type="workout_next_step",
            title=first_line[:160],
            body=body,
            evidence={"comparison": comparison, "goal_id": history_evidence["goal_id"]},
            confidence="low",
        )
        result = self.store.get_recommendation(rec_id)
        assert result is not None
        return result

    def ask(self, question: str) -> tuple[str, dict[str, Any]]:
        context = self.build_context()
        user_prompt = (
            "MEMORY_CONTEXT_JSON:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
            + "\n\nUSER_QUESTION:\n"
            + question
        )
        answer = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        return answer, context
