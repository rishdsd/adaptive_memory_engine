from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from .db import MemoryStore
from .ollama_client import OllamaClient
from .workout_data import WorkoutRepository, normalize_exercise_name

PLANNER_SYSTEM_PROMPT = """You are a query planner for a personal workout-history application.
Return exactly one JSON object. Do not answer the user's question and do not invent facts.
Use the CURRENT USER MESSAGE to identify the request. If the user is clearly continuing a query,
use the supplied conversation context if present. Do not turn a history question into a log request.

Supported intents:
- query: asks about already-recorded workout history
- log_workout: says a workout was completed and asks to record/remember it
- unsupported: neither of the above or insufficient information to classify safely

Supported query_type values:
- best_set: the highest-rep recorded set for an exercise (unless the user explicitly asks for another metric)
- latest_set: the latest recorded session/set for an exercise
- exercise_history: recorded sets/sessions for one exercise over time; includes paraphrases such as "story", "track record", "past sets", or "how has it gone"
- session_exercises: exercises in a specified session/date
- list_exercises: distinct exercises across recorded history
- list_sessions: recorded workout sessions

Fields:
- exercise_name: extract the exercise as the user named it, or null if not identifiable
- session_label/session_date: only when specified or clearly inferable from context
- load_type: "bodyweight", "assisted", or "weighted" only when the user explicitly requests that type; otherwise null
- include_dates: false only when the user explicitly asks to omit dates; true otherwise
- clarification: a short question only if an essential parameter is genuinely missing; otherwise empty string
- log: for log_workout only, extract explicit workout values. Never invent reps/load/date. Omitted date is null.

Important interpretation rules:
- Synonyms and informal wording should map to the closest supported query type. Do not require exact keywords.
- "best", "highest reps", "most reps", "PR" map to best_set.
- "latest", "last time", "most recent" map to latest_set.
- "history", "story", "past sets", "over time", "progression" map to exercise_history.
- If a history query names an exercise, do not ask the user to name it again.
- Do not infer bodyweight/assisted/weighted when not stated. The application applies its own transparent default policy.
- Do not treat an exercise the user plans to do as completed.
- Do not produce SQL or answer from memory; Python will query the database.

Required JSON shape:
{
  "intent": "query|log_workout|unsupported",
  "query_type": "best_set|latest_set|exercise_history|session_exercises|list_exercises|list_sessions|null",
  "exercise_name": "string or null",
  "session_label": "string or null",
  "session_date": "YYYY-MM-DD or null",
  "load_type": "bodyweight|assisted|weighted|null",
  "include_dates": true,
  "log": {
    "session_date": "YYYY-MM-DD or null",
    "session_label": "string or null",
    "exercises": [{"name":"string", "sets":[{"reps":12,"load_kg":25}]}],
    "notes": []
  },
  "clarification": "short question or empty string"
}

For logging, supported set keys are reps, load ("bodyweight"), load_kg, load_kg_per_hand,
assistance_kg, machine_level, machine_setting. Include only explicitly stated values. A set can have
multiple supported fields. If essential information is missing, return intent=unsupported and ask a
short clarification. Return valid JSON only.
"""


class StageOneChat:
    """Stage 1: LLM interprets; deterministic code retrieves, computes, and formats factual answers."""

    def __init__(self, store: MemoryStore, repository: WorkoutRepository, llm: OllamaClient) -> None:
        self.store = store
        self.repository = repository
        self.llm = llm

    def _canonical_exercise_name(self, name: str | None) -> str | None:
        """Resolve an LLM-extracted name against names actually present in the database."""
        if not name or not str(name).strip():
            return None
        requested = normalize_exercise_name(str(name))
        known = self.repository.query({"query_type": "list_exercises"}).get("exercises", [])
        exact = [str(item) for item in known if normalize_exercise_name(str(item)) == requested]
        if len(exact) == 1:
            return exact[0]
        # Accept a unique containment match, but avoid guessing when multiple names fit.
        nearby = [str(item) for item in known
                  if requested and (requested in normalize_exercise_name(str(item))
                                    or normalize_exercise_name(str(item)) in requested)]
        if len(nearby) == 1:
            return nearby[0]
        return str(name).strip()

    @staticmethod
    def _validate_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
        """Validate planner output before it can reach repository/query or write logic."""
        intents = {"query", "log_workout", "unsupported"}
        query_types = {"best_set", "latest_set", "exercise_history", "session_exercises", "list_exercises", "list_sessions"}
        if plan.get("intent") not in tuple(intents):
            return None
        if plan.get("intent") == "query" and plan.get("query_type") not in tuple(query_types):
            return None
        if plan.get("load_type") not in (None, "bodyweight", "assisted", "weighted"):
            plan["load_type"] = None
        if not isinstance(plan.get("include_dates"), bool):
            plan["include_dates"] = True
        if not isinstance(plan.get("exercise_name"), (str, type(None))):
            plan["exercise_name"] = None
        return plan

    def process(self, message: str) -> dict[str, Any]:
        # The model interprets language; this class validates its plan. No phrase-specific regex routing.
        raw = self.llm.chat_json(
            PLANNER_SYSTEM_PROMPT,
            "CURRENT_DATE: " + date.today().isoformat() + "\nUSER_MESSAGE:\n" + message,
        )
        try:
            plan = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"mode": "clarification", "answer": "I couldn't reliably interpret that. Could you rephrase it?"}
        if not isinstance(plan, dict):
            return {"mode": "clarification", "answer": "I couldn't reliably interpret that. Could you rephrase it?"}
        plan = self._validate_plan(plan)
        if plan is None:
            return {"mode": "clarification", "answer": "I couldn't reliably map that request. Could you rephrase it?"}

        intent = plan.get("intent")
        if intent == "log_workout":
            log = plan.get("log") or {}
            # Preserve an explicitly stated bodyweight load if the model omitted it, without
            # assuming that all pull-ups/push-ups are bodyweight.
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
            query_type = plan["query_type"]
            if query_type in {"best_set", "latest_set", "exercise_history"}:
                canonical = self._canonical_exercise_name(plan.get("exercise_name"))
                if not canonical:
                    return {"mode": "clarification", "answer": "Which exercise should I look up?"}
                plan["exercise_name"] = canonical
            # User experience policy: without an explicit load filter, a best-set query
            # reports the best set in each load category separately rather than comparing
            # incomparable bodyweight, assisted, and weighted sets as one number.
            if query_type == "best_set" and plan.get("load_type") is None:
                plan["group_by_load"] = True
            plan["include_dates"] = bool(plan.get("include_dates", True))
            result = self.repository.query(plan)
            result["include_dates"] = plan["include_dates"]
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
        has_load = any(set_data.get(key) is not None for key in
                       ("load", "load_kg", "load_kg_per_hand", "assistance_kg"))
        if set_data.get("reps") is not None and not has_load:
            parts.append("load not recorded")
        return ", ".join(parts) if parts else "measurements not recorded"

    @classmethod
    def _format_result(cls, result: dict[str, Any]) -> str:
        kind = result.get("query_type")
        if kind == "best_set":
            grouped = result.get("best_sets_by_load_type")
            if grouped is not None:
                if not grouped:
                    return f"I couldn't find recorded sets for {result.get('exercise_name')} matching those filters."
                lines = [f"Your best recorded {result.get('exercise_name')} set by load type:"]
                for category, best in grouped.items():
                    date_text = f" on {best['date']}" if result.get("include_dates", True) else ""
                    lines.append(f"- {category.replace('_', ' ')}: {cls._format_set(best)}{date_text}")
                lines.append("These categories are shown separately because assisted, bodyweight, and weighted sets are not directly comparable.")
                return "\n".join(lines)
            best = result.get("best_set")
            if not best:
                return f"I couldn't find recorded sets for {result.get('exercise_name')} matching those filters."
            date_text = f" on {best['date']}" if result.get("include_dates", True) else ""
            return (f"Your best recorded set for {best['exercise_name']} was {cls._format_set(best)}{date_text}. "
                    "This is the best among the matching records in the database, not a live fitness assessment.")
        if kind == "latest_set":
            sets = result.get("sets") or []
            if not sets:
                return f"I couldn't find recorded sets for {result.get('exercise_name')}."
            if result.get("include_dates", True):
                lines = [f"Your latest recorded {result['exercise_name']} session is {result['latest_date']}:"]
            else:
                lines = [f"Your latest recorded {result['exercise_name']} session:"]
            lines.extend(f"Set {s.get('set_number', i)}: {cls._format_set(s)}" for i, s in enumerate(sets, 1))
            return "\n".join(lines)
        if kind == "exercise_history":
            records = result.get("records") or []
            if not records:
                return f"I couldn't find recorded sets for {result.get('exercise_name')}."
            lines = [f"Recorded history for {result.get('exercise_name')}:"]
            if result.get("include_dates", True):
                lines.extend(f"{r['date']}: Set {r.get('set_number', '?')} — {cls._format_set(r)}" for r in records)
            else:
                lines.extend(f"Set {r.get('set_number', '?')} — {cls._format_set(r)}" for r in records)
            return "\n".join(lines)
        if kind in {"session_exercises", "list_exercises"}:
            exercises = result.get("exercises") or []
            if not exercises:
                return "I couldn't find any recorded exercises matching those filters."
            scope = result.get("filters", {})
            if scope.get("session_label"):
                intro = f"Exercises found in recorded sessions matching '{scope['session_label']}':"
            elif scope.get("session_date") and result.get("include_dates", True):
                intro = f"Exercises recorded on {scope['session_date']}:"
            elif scope.get("session_date"):
                intro = "Exercises recorded in the requested session:"
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
                label = session['session_label'] or 'label not recorded'
                if result.get("include_dates", True):
                    lines.append(f"{session['date']} ({label}): {names or 'no exercises recorded'}")
                else:
                    lines.append(f"{label}: {names or 'no exercises recorded'}")
            return "\n".join(lines)
        return "I couldn't answer that from the supported workout history queries."
