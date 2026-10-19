from __future__ import annotations

import json
from typing import Any

from .db import MemoryStore
from .metrics import calculate_goal_progress
from .ollama_client import OllamaClient

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
