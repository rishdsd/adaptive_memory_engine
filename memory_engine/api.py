
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .agent import MemoryAgent
from .db import MemoryStore
from .metrics import calculate_goal_progress
from .ollama_client import OllamaClient, OllamaError

DB_PATH = os.environ.get("MEMORY_DB", "data/memory.db")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

store = MemoryStore(DB_PATH)
app = FastAPI(title="Personal AI Progress Coach", version="0.2.0")


def get_agent() -> MemoryAgent:
    return MemoryAgent(
        store,
        OllamaClient(OLLAMA_MODEL, OLLAMA_URL),
    )


class GoalRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    metric_name: str = Field(min_length=1, max_length=100)
    target_value: float = Field(gt=0, allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=50)
    aggregation: Literal["sum", "latest"] = "sum"


class ProgressRequest(BaseModel):
    goal_id: int = Field(gt=0)
    value: float = Field(ge=0, allow_inf_nan=False)
    note: str = Field(default="", max_length=1000)
    occurred_at: datetime | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class FeedbackRequest(BaseModel):
    status: Literal["accepted", "rejected", "modified"]
    note: str = Field(default="", max_length=2000)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "personal-ai-progress-coach"}


@app.get("/goals")
def get_goals(active_only: bool = True) -> list[dict]:
    return store.list_goals(active_only=active_only)


@app.post("/goals", status_code=201)
def create_goal(request: GoalRequest) -> dict:
    try:
        goal_id = store.create_goal(
            request.title,
            request.metric_name,
            request.target_value,
            request.unit,
            request.aggregation,
        )
        return store.get_goal(goal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/goals/{goal_id}/progress")
def get_goal_progress(goal_id: int) -> dict:
    if store.get_goal(goal_id) is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return calculate_goal_progress(store, goal_id)


@app.get("/goals/{goal_id}/events")
def get_goal_events(
    goal_id: int,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    if store.get_goal(goal_id) is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return store.get_events(goal_id, limit=limit)


@app.post("/progress", status_code=201)
def record_progress(request: ProgressRequest) -> dict:
    if store.get_goal(request.goal_id) is None:
        raise HTTPException(status_code=404, detail="Goal not found")

    occurred_at = (
        request.occurred_at.isoformat()
        if request.occurred_at is not None
        else None
    )

    try:
        event_id = store.record_progress(
            goal_id=request.goal_id,
            value=request.value,
            note=request.note,
            occurred_at=occurred_at,
            source="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "event_id": event_id,
        "goal_id": request.goal_id,
        "progress": calculate_goal_progress(store, request.goal_id),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    try:
        answer, context = get_agent().ask(request.question)
        return {"answer": answer, "context": context}
    except OllamaError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI model unavailable: {exc}",
        ) from exc


@app.get("/recommendations")
def get_recommendations(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    return store.list_recommendations(limit=limit)


@app.post("/recommendations/pullup", status_code=201)
def recommend_pullup(goal_id: int | None = None) -> dict:
    try:
        return get_agent().recommend_pullup_next_step(goal_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommendations/{recommendation_id}/feedback")
def recommendation_feedback(
    recommendation_id: int,
    request: FeedbackRequest,
) -> dict:
    try:
        store.record_recommendation_feedback(
            recommendation_id,
            request.status,
            request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return store.get_recommendation(recommendation_id)
