# Adaptive Memory Engine — Iteration 1

A local-first prototype for testing whether persistent memory plus deterministic metrics can help a small local LLM answer questions and make better recommendations.

## Current scope

- SQLite persistence for goals and append-only progress events
- Deterministic progress calculation (`sum` or `latest` aggregation)
- Ollama chat integration using Python's standard library
- An initial memory context builder and CLI
- Unit tests for core memory and metric behavior

This is intentionally a small vertical slice. Semantic/vector retrieval, recommendation feedback, dashboard, reminders, and task execution are later milestones.

## Requirements

- Python 3.10+
- Ollama installed and running locally
- Model `qwen3.5:2b` available in Ollama (or set `OLLAMA_MODEL` to your installed model)
- No third-party Python packages required for this iteration

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v
ollama list
```

If needed, start Ollama and pull the model:

```bash
ollama serve
ollama pull qwen3.5:2b
```

If Ollama is already running as a Mac application, you usually do not need to run `ollama serve` separately.

## Try the vertical slice

Create a cumulative goal:

```bash
python -m memory_engine.cli create-goal "Study DSA" --metric study_hours --target 12 --unit hours --aggregation sum
```

The command prints the goal ID. Use that ID below (shown as `1` here):

```bash
python -m memory_engine.cli log-progress 1 1.5 --note "Arrays practice"
python -m memory_engine.cli log-progress 1 2 --note "Sliding window practice"
python -m memory_engine.cli progress 1
python -m memory_engine.cli ask "How much have I studied, and what would be a sensible next step?"
```

Restart the terminal or run the commands in a new shell, then ask again. The SQLite database defaults to `data/memory.db` and persists across runs.

For a latest-value goal (such as a measurement), use `--aggregation latest` and record each new measurement as an event.

## Configuration

```bash
export MEMORY_DB="$PWD/data/memory.db"
export OLLAMA_MODEL="qwen3.5:2b"
export OLLAMA_URL="http://localhost:11434"
```

The database path can also be passed using the global `--db` option before the subcommand, for example `python -m memory_engine.cli --db ./test.db list-goals`.

## Current limitations

- The first context builder includes all active goals; it does not yet rank or retrieve memories semantically.
- The agent is instructed to use the provided context, but model output is not a proof of correctness. Verify claims against the printed evidence context.
- Event timestamps are accepted as ISO strings; strict timezone normalization and correction workflows are not yet implemented.
- No autonomous actions or reminders are executed in this iteration.

## Stage 1: accurate workout-history chat

Stage 1 uses the local LLM only to turn a natural-language message into a constrained JSON intent/query plan. Python validates that plan, reads the SQLite history, computes any result, and formats factual answers deterministically. The model does not generate arbitrary SQL and does not write directly to the database.

The existing `events.payload_json` remains the canonical append-only record. On startup, `memory_engine.workout_data.WorkoutRepository` creates normalized `workout_sessions`, `workout_exercises`, and `exercise_sets` tables and idempotently indexes any existing `gym_session_log` events. Existing records are not deleted or rewritten.

Run the API and open the local chat UI at `http://127.0.0.1:8000/`:

```bash
uvicorn memory_engine.api:app --reload
```

You can also call the API directly. Ask a history question (the local Ollama model must be running):

```bash
curl -s http://127.0.0.1:8000/chat \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"What is my best bodyweight pull-up set?"}'
```

To log a completed workout, send the natural-language message to `/chat`. If it returns `mode: log_confirmation_required`, review the `pending_log` and explicitly confirm it through `/chat/confirm-log`. Only then is a canonical event written and indexed:

```bash
curl -s http://127.0.0.1:8000/chat/confirm-log \\
  -H 'Content-Type: application/json' \\
  -d '{"session_date":"2026-10-19","session_label":"back","exercises":[{"name":"Seated row","sets":[{"reps":12,"load_kg":25},{"reps":12,"load_kg":25},{"reps":12,"load_kg":25}]}],"notes":[]}'
```

Then ask `/chat` again to retrieve the newly stored entry. The two-step write flow is intentional: model-extracted data is previewed before persistence. Stage 1 currently supports workout-history queries and workout logging; broader goal/event chat queries and Stage 2 recommendations are not yet part of this vertical slice.

### AI engineering decision

This is constrained tool selection, not MCP yet. The LLM produces a small JSON plan from an allowlisted schema; application code validates and executes known operations. MCP is a protocol for exposing tools/resources to compatible AI clients. We should add an MCP adapter only if we later want this same memory/query capability available to multiple MCP-capable clients. It would wrap these existing safe functions, not replace the database or accuracy tests.

### Stage 1 tests

```bash
python -m unittest discover -s tests -v
```

Tests use a fake planner, so correctness tests do not require Ollama to be running. Live-model evaluation is a separate step because the model can still misclassify or mis-extract a user message.
