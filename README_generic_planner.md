# Generic planner iteration 1

## What changed

- Removed phrase-specific regex routing from the chat query path. Every natural-language request is interpreted by the planner prompt, then validated in Python.
- Added `include_dates` as a presentation preference rather than a new query operation.
- Added load-aware best-set grouping: if the user asks for a best set without specifying bodyweight/assisted/weighted, the application reports the best recorded set in each category separately. This avoids comparing unlike sets as one score.
- Added an explicit `load_not_recorded` category. A set without a recorded load is not silently assumed to be bodyweight.
- Extended repository filtering to support weighted sets.
- Added a labelled planner evaluation set and a standard-library-only Ollama evaluation script.

## Apply

Back up your current files first. From the project root:

```bash
cp memory_engine/stage1_chat.py memory_engine/stage1_chat.py.before_generic
cp memory_engine/workout_data.py memory_engine/workout_data.py.before_generic
```

Copy the supplied `stage1_chat.py` and `workout_data.py` into the project's `memory_engine/` directory. Keep `data/memory.db` unchanged.

## Run unit tests

Copy `test_generic_planner.py` into the project root and run:

```bash
python -m pytest -q test_generic_planner.py
```

The fake-model tests verify Python validation and formatting; they do not measure actual LLM quality.

## Run the model evaluation

Copy `planner_eval.json` and `evaluate_planner.py` into the project root. Ensure Ollama is running and the model is available, then run:

```bash
python evaluate_planner.py --model qwen3.5:2b
python evaluate_planner.py --model qwen3.5:4b
```

Each run writes a model-specific `planner_eval_results_*.json` file. Compare exact-case and field accuracy. The evaluation calls the same `PLANNER_SYSTEM_PROMPT` used by the application.

## Scope and limitations

- This first evaluation is single-turn. Follow-up/conversation-state support is a separate iteration; do not infer that context persistence is implemented.
- The labelled set covers existing query types and a few logging/unsupported cases. It intentionally does not test new capabilities such as comparing progress trends, because the repository doesn't yet expose those as supported query operations.
- Review any expected labels that disagree with your product policy before treating the score as definitive.
- Do not add more historical data yet. First use the current pull-up records to measure intent, exercise extraction, load filter, and presentation handling. Then broaden the evaluation to push-ups and other exercise types.
