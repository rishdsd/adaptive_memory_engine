
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import MemoryAgent
from .db import MemoryStore
from .metrics import calculate_goal_progress
from .ollama_client import OllamaClient, OllamaError
from .stage1_chat import StageOneChat
from .workout_data import WorkoutRepository

DB_PATH = os.environ.get("MEMORY_DB", "data/memory.db")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

store = MemoryStore(DB_PATH)
workout_repository = WorkoutRepository(DB_PATH)  # idempotently indexes existing workout payloads
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


class WorkoutLogConfirmation(BaseModel):
    session_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    session_label: str = Field(default="", max_length=100)
    exercises: list[dict]
    notes: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    status: Literal["accepted", "rejected", "modified"]
    note: str = Field(default="", max_length=2000)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_home() -> str:
    """A tiny local chat UI for Stage 1; all persistence still happens through validated API routes."""
    return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal Memory — Stage 1</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#202124;background:#f6f7f9}body{max-width:820px;margin:32px auto;padding:0 16px}
main{background:white;border:1px solid #ddd;border-radius:16px;padding:20px;box-shadow:0 4px 18px #0000000a}
h1{font-size:1.35rem;margin:0 0 6px}.sub{color:#666;font-size:.92rem;margin:0 0 18px}#messages{display:flex;flex-direction:column;gap:12px;min-height:280px;max-height:58vh;overflow:auto;padding:4px 0 18px}
.msg{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px 14px;border-radius:12px;max-width:92%;line-height:1.45}.user{align-self:flex-end;background:#e8f0fe}.assistant{align-self:flex-start;background:#f0f1f3}.pending{border:1px solid #d7a33c;background:#fff9e8;padding:12px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}.controls{display:flex;gap:8px;margin-top:12px}input{flex:1;min-width:0;padding:12px;border:1px solid #bbb;border-radius:10px;font:inherit}button{border:0;border-radius:10px;padding:11px 15px;background:#202124;color:white;font:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.confirm{margin-top:10px;background:#176b3a}.small{font-size:.8rem;color:#666;margin-top:14px}</style></head>
<body><main><h1>Personal Memory — Stage 1</h1><p class="sub">Ask about recorded workouts, or log a completed session. Historical answers come from SQLite.</p>
<div id="messages" aria-live="polite"><div class="msg assistant">Hi! Ask about your recorded workout history, or tell me about a workout you completed. New logs are previewed before saving.</div></div>
<form id="form"><div class="controls"><input id="message" placeholder="e.g. What's my best bodyweight pull-up set?" autocomplete="off" required><button id="send" type="submit">Send</button></div></form>
<p class="small">Stage 1 only uses recorded workout history. Recommendations and external knowledge are not enabled.</p></main>
<script>
const messages=document.getElementById('messages'),form=document.getElementById('form'),input=document.getElementById('message'),send=document.getElementById('send');
function bubble(text,kind='assistant'){const el=document.createElement('div');el.className='msg '+kind;el.textContent=text;messages.appendChild(el);messages.scrollTop=messages.scrollHeight;return el;}
async function post(url,data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const body=await r.json();if(!r.ok)throw new Error(body.detail||'Request failed');return body;}
form.addEventListener('submit',async e=>{e.preventDefault();const question=input.value.trim();if(!question)return;bubble(question,'user');input.value='';send.disabled=true;
try{const result=await post('/chat',{question});const answer=bubble(result.answer||'No answer returned.');if(result.mode==='log_confirmation_required'&&result.pending_log){const preview=document.createElement('div');preview.className='pending';preview.textContent=JSON.stringify(result.pending_log,null,2);answer.appendChild(document.createElement('br'));answer.appendChild(preview);const button=document.createElement('button');button.className='confirm';button.textContent='Confirm and save this workout';answer.appendChild(button);button.addEventListener('click',async()=>{button.disabled=true;try{const saved=await post('/chat/confirm-log',result.pending_log);bubble('Saved to your history. Event ID: '+saved.event_id+'\nYou can now ask me to retrieve this record.');button.textContent='Saved';}catch(err){bubble('Could not save: '+err.message);button.disabled=false;}});}}
catch(err){bubble('Error: '+err.message);}finally{send.disabled=false;input.focus();}});
</script></body></html>"""


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


def get_stage1_chat() -> StageOneChat:
    return StageOneChat(store, workout_repository, OllamaClient(OLLAMA_MODEL, OLLAMA_URL))


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Stage 1 chat: LLM parses intent; application code reads or prepares a validated write."""
    try:
        return get_stage1_chat().process(request.question)
    except OllamaError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI model unavailable: {exc}",
        ) from exc


@app.post("/chat/confirm-log", status_code=201)
def confirm_workout_log(request: WorkoutLogConfirmation) -> dict:
    """Commit a previously previewed log only after explicit user confirmation."""
    payload = {
        "session_date": request.session_date,
        "session_label": request.session_label,
        "exercises": request.exercises,
        "notes": request.notes,
    }
    validated = get_stage1_chat()._prepare_log(payload, "")
    if validated.get("mode") != "log_confirmation_required":
        raise HTTPException(status_code=422, detail=validated.get("answer", "Invalid workout log"))
    pending = validated["pending_log"]
    try:
        return {"status": "saved", **workout_repository.create_workout_event(
            store=store,
            session_date=pending["session_date"],
            session_label=pending["session_label"],
            exercises=pending["exercises"],
            notes=pending["notes"],
        )}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
