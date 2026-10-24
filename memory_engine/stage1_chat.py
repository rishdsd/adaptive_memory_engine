from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from .db import MemoryStore
from .ollama_client import OllamaClient
from .workout_data import WorkoutRepository, normalize_exercise_name

PLANNER_SYSTEM_PROMPT = """You are a strict intent and data extraction component for a personal-history chatbot.
Use only the user's message to produce one JSON object matching the requested schema. Do not answer
in prose and do not invent historical facts. Choose intent=query for questions about recorded history,
intent=log_workout when the user says they completed a workout and wants it remembered, otherwise
intent=unsupported. For a query, choose query_type from: best_set, latest_set, exercise_history,
session_exercises, list_exercises, list_sessions. `session_exercises` means list unique exercises
actually recorded in matching sessions. `list_exercises` means list unique recorded exercises generally.
Use `load_type=bodyweight` only when explicitly asking for bodyweight; use `assisted` only when asking
for assisted sets. Do not treat a planned exercise as completed. For logging, extract only explicit
values, use ISO YYYY-MM-DD for an explicitly stated date, and set omitted date to null. Do not infer
reps, weights, or exercises. Return exactly the JSON object, with all top-level fields present.

Required object shape:
{
  "intent": "query|log_workout|unsupported",
  "query_type": "best_set|latest_set|exercise_history|session_exercises|list_exercises|list_sessions|null",
  "exercise_name": "string or null",
  "session_label": "string or null",
  "session_date": "YYYY-MM-DD or null",
  "load_type": "bodyweight|assisted|null",
  "log": {
    "session_date": "YYYY-MM-DD or null",
    "session_label": "string or null",
    "exercises": [{"name":"string", "sets":[{"reps":12,"load_kg":25}]}],
    "notes": []
  },
  "clarification": "short question or empty string"
}
For log sets, use these supported measurement keys where explicitly present: reps, load (e.g. bodyweight),
load_kg, load_kg_per_hand, assistance_kg, machine_level, machine_setting. A set may contain more than
one supported measurement. If the user did not provide a usable exercise name or any set details, set
intent=unsupported and ask a clarification. Do not produce SQL.
"""


class StageOneChat:
    """Stage 1: LLM interprets; deterministic code retrieves, computes, and formats factual answers."""

    def __init__(self, store: MemoryStore, repository: WorkoutRepository, llm: OllamaClient) -> None:
        self.store = store
        self.repository = repository
        self.llm = llm

    def _deterministic_query_plan(self, message: str) -> dict[str, Any] | None:
        """Handle clear history questions without asking the LLM to infer obvious intent."""
        text = message.casefold()
        normalized_message = normalize_exercise_name(text)

        # Match against exercise names actually present in the user's history.
        known = self.repository.query({"query_type": "list_exercises"}).get("exercises", [])
        candidates = []
        for name in known:
            key = normalize_exercise_name(str(name))
            if key and key in normalized_message:
                candidates.append((len(key), str(name)))
        if not candidates:
            return None

        # Prefer the longest match (e.g. "bench press" over "press").
        exercise_name = max(candidates, key=lambda item: item[0])[1]

        if re.search(r"\b(best|highest|maximum|max|most reps|personal record|record)\b", text):
            query_type = "best_set"
        elif re.search(r"\b(latest|last|most recent)\b", text):
            query_type = "latest_set"
        elif re.search(r"\b(history|all sets|every set|progress|over time)\b", text):
            query_type = "exercise_history"
        else:
            return None

        if re.search(r"\bbody\s*weight\b", text):
            load_type = "bodyweight"
        elif re.search(r"\b(assisted|assistance)\b", text):
            load_type = "assisted"
        else:
            load_type = None

        return {
            "intent": "query",
            "query_type": query_type,
            "exercise_name": exercise_name,
            "session_label": None,
            "session_date": None,
            "load_type": load_type,
        }

    def process(self, message: str) -> dict[str, Any]:
        # Common, unambiguous history questions should not depend on an LLM classification.
        plan = self._deterministic_query_plan(message)
        if plan is None:
            raw = self.llm.chat_json(
                PLANNER_SYSTEM_PROMPT,
                "CURRENT_DATE: " + date.today().isoformat() + "\nUSER_MESSAGE:\n" + message,
            )
            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                return {"mode": "clarification", "answer": "I couldn't reliably interpret that. Could you rephrase it?"}
        if not isinstance(plan, dict):
            return {"mode": "clarification", "answer": "I couldn't reliably interpret that. Could you rephrase it?"}

        intent = plan.get("intent")
        if intent == "log_workout":
            log = plan.get("log") or {}
            # Preserve an explicitly stated bodyweight load if the LLM omitted it.
            # Restrict this repair to a single-exercise log to avoid applying it to
            # unrelated exercises in a multi-exercise workout.
            exercises = log.get("exercises")
            if (re.search(r"\bbody\s*weight\b", message, re.IGNORECASE)
                    and isinstance(exercises, list) and len(exercises) == 1
                    and isinstance(exercises[0], dict)):
                sets = exercises[0].get("sets")
                if isinstance(sets, list):
                    for set_data in sets:
                        if not isinstance(set_data, dict):
                            continue
                        load_fields = {"load", "load_kg", "load_kg_per_hand", "assistance_kg"}
                        if not any(set_data.get(key) is not None for key in load_fields):
                            set_data["load"] = "bodyweight"
            return self._prepare_log(log, plan.get("clarification") or "")
        if intent == "query":
            query_type = plan.get("query_type")
            if query_type not in {"best_set", "latest_set", "exercise_history", "session_exercises", "list_exercises", "list_sessions"}:
                return {"mode": "clarification", "answer": "I couldn't map that to a supported history query. Could you rephrase it?"}
            if query_type in {"best_set", "latest_set", "exercise_history"} and not str(plan.get("exercise_name") or "").strip():
                return {"mode": "clarification", "answer": "Which exercise should I look up?"}
            result = self.repository.query(plan)
            return {"mode": "history_answer", "answer": self._format_result(result), "result": result,
                    "source": "sqlite_workout_history"}
        clarification = str(plan.get("clarification") or "").strip()
        return {"mode": "clarification", "answer": clarification or
                "I can answer questions about recorded workout history or help log a completed workout. What would you like to look up or record?"}

    def _prepare_log(self, log: dict[str, Any], clarification: str) -> dict[str, Any]:
        exercises = log.get("exercises")
        if not isinstance(exercises, list) or not exercises:
            return {"mode": "clarification", "answer": clarification or
                    "Which exercise did you complete, and what sets or measurements should I record?"}
        cleaned = []
        allowed = {"reps", "load", "load_kg", "load_kg_per_hand", "assistance_kg", "machine_level", "machine_setting"}
        for exercise in exercises:
            if not isinstance(exercise, dict) or not str(exercise.get("name") or "").strip():
                return {"mode": "clarification", "answer": "I need the exercise name before I can prepare this log."}
            sets = exercise.get("sets")
            if not isinstance(sets, list) or not sets:
                return {"mode": "clarification", "answer": f"How many sets or what measurements should I record for {exercise['name']}?"}
            clean_sets = []
            for set_data in sets:
                if not isinstance(set_data, dict):
                    return {"mode": "clarification", "answer": "One of the set entries wasn't structured clearly. Please restate that set."}
                unknown_fields = set(set_data) - allowed
                if unknown_fields:
                    return {"mode": "clarification", "answer":
                            "I found measurement fields I don't yet support (" + ", ".join(sorted(unknown_fields)) +
                            "). Please clarify those values before I save the workout."}
                selected = {k: v for k, v in set_data.items() if k in allowed and v is not None}
                if not selected:
                    return {"mode": "clarification", "answer": f"I couldn't identify reps, load, assistance, or machine settings for {exercise['name']}. Please clarify the set details."}
                if "reps" in selected and (not isinstance(selected["reps"], int) or selected["reps"] < 0):
                    return {"mode": "clarification", "answer": "The reps value must be a non-negative whole number. Please clarify it."}
                if "load" in selected and selected["load"] != "bodyweight":
                    return {"mode": "clarification", "answer": "For load, please specify bodyweight or give a numeric load in kilograms."}
                for key in allowed - {"load"}:
                    if key in selected and (not isinstance(selected[key], (int, float)) or selected[key] < 0):
                        return {"mode": "clarification", "answer": f"The value for {key} must be non-negative. Please clarify it."}
                clean_sets.append(selected)
            cleaned.append({"name": str(exercise["name"]).strip(), "sets": clean_sets})

        session_date = log.get("session_date") or date.today().isoformat()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(session_date)):
            return {"mode": "clarification", "answer": "What date should I attach to this workout? Please use YYYY-MM-DD."}
        try:
            date.fromisoformat(session_date)
        except ValueError:
            return {"mode": "clarification", "answer": "That date doesn't look valid. Please provide it as YYYY-MM-DD."}
        prepared = {"session_date": session_date, "session_label": str(log.get("session_label") or "").strip(),
                    "exercises": cleaned, "notes": log.get("notes") if isinstance(log.get("notes"), list) else []}
        date_note = " (using today's date because no date was specified)" if not log.get("session_date") else ""
        return {"mode": "log_confirmation_required",
                "answer": f"I parsed this workout{date_note}. Please confirm before I save it.",
                "pending_log": prepared}

    @staticmethod
    def _format_set(set_data: dict[str, Any]) -> str:
        parts = []
        if set_data.get("reps") is not None:
            parts.append(f"{set_data['reps']} reps")
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
        return ", ".join(parts) if parts else "measurements not recorded"

    @classmethod
    def _format_result(cls, result: dict[str, Any]) -> str:
        kind = result.get("query_type")
        if kind == "best_set":
            best = result.get("best_set")
            if not best:
                return f"I couldn't find recorded sets for {result.get('exercise_name')} matching those filters."
            return (f"Your best recorded set for {best['exercise_name']} was {cls._format_set(best)} on {best['date']}. "
                    "This is the best among the matching records in the database, not a live fitness assessment.")
        if kind == "latest_set":
            sets = result.get("sets") or []
            if not sets:
                return f"I couldn't find recorded sets for {result.get('exercise_name')}."
            lines = [f"Your latest recorded {result['exercise_name']} session is {result['latest_date']}:"]
            lines.extend(f"Set {s.get('set_number', i)}: {cls._format_set(s)}" for i, s in enumerate(sets, 1))
            return "\n".join(lines)
        if kind == "exercise_history":
            records = result.get("records") or []
            if not records:
                return f"I couldn't find recorded sets for {result.get('exercise_name')}."
            lines = [f"Recorded history for {result.get('exercise_name')}:"]
            lines.extend(f"{r['date']}: Set {r.get('set_number', '?')} — {cls._format_set(r)}" for r in records)
            return "\n".join(lines)
        if kind in {"session_exercises", "list_exercises"}:
            exercises = result.get("exercises") or []
            if not exercises:
                return "I couldn't find any recorded exercises matching those filters."
            scope = result.get("filters", {})
            if scope.get("session_label"):
                intro = f"Exercises found in recorded sessions matching '{scope['session_label']}':"
            elif scope.get("session_date"):
                intro = f"Exercises recorded on {scope['session_date']}:"
            else:
                intro = "Exercises present in your recorded workout history:"
            return intro + "\n- " + "\n- ".join(exercises)
        if kind == "list_sessions":
            sessions = result.get("sessions") or []
            if not sessions:
                return "There are no workout sessions in the indexed history yet."
            lines = [f"Found {len(sessions)} recorded workout session(s):"]
            for session in sessions:
                names = ", ".join(ex["name"] for ex in session["exercises"])
                lines.append(f"{session['date']} ({session['session_label'] or 'label not recorded'}): {names or 'no exercises recorded'}")
            return "\n".join(lines)
        return "I couldn't answer that from the supported workout history queries."
