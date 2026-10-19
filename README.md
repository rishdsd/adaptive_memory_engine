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
