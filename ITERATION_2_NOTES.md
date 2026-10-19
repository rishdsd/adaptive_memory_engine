# Adaptive Memory Engine — Iteration 2

## What's new

- Deterministic gym-history rendering in `memory_engine/workout.py`. Python renders each set explicitly, separating reps, load, assistance, machine levels, and machine settings.
- Pull-up comparison based on the recorded 2026-09-08 and 2026-09-23 entries. The comparison includes a caution that two partial records do not establish an overall strength trend.
- SQLite persistence for recommendations, their supporting evidence, confidence, status, and user feedback.
- CLI commands for gym history, creating a pull-up next-step recommendation, listing recommendations, and recording feedback.
- Regression tests for reps-versus-sets, machine settings versus kilograms, cautious comparison, and feedback persistence.

## Commands

Run from the project root after activating your virtual environment:

```bash
python -m unittest discover -s tests -v
python -m memory_engine.cli gym-history
python -m memory_engine.cli recommend-pullups
python -m memory_engine.cli list-recommendations
python -m memory_engine.cli recommendation-feedback 1 accepted --note "Useful for my next workout"
```

If the gym logs are in a non-default goal, add `--goal-id <id>` to `gym-history` or `recommend-pullups`.

The recommendation command calls the configured local Ollama model and persists the result. The first recommendation will normally have ID 1 in a fresh database; check `list-recommendations` for the actual ID.

## Safety and evidence rules

- The application, not the LLM, formats factual exercise sets.
- Machine levels/settings are not treated as kilograms.
- Missing reps or load remain missing.
- The pull-up comparison is explicitly low-confidence and should not be interpreted as a strength diagnosis or definitive trend.
- Recommendation feedback is stored for a future adaptation loop; this version does not yet train or automatically change future recommendations based on feedback.
