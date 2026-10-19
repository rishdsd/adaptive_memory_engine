from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import MemoryAgent
from .db import MemoryStore
from .metrics import calculate_goal_progress
from .ollama_client import OllamaClient, OllamaError
from .workout import render_gym_history


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory-engine", description="Local-first Adaptive Memory Engine prototype")
    parser.add_argument("--db", default=os.environ.get("MEMORY_DB", "data/memory.db"), help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-goal", help="Create a goal")
    create.add_argument("title")
    create.add_argument("--metric", required=True, help="Metric name, e.g. study_hours")
    create.add_argument("--target", required=True, type=float)
    create.add_argument("--unit", required=True, help="Unit, e.g. hours")
    create.add_argument("--aggregation", choices=["sum", "latest"], default="sum")

    sub.add_parser("list-goals", help="List goals")

    gym_history = sub.add_parser("gym-history", help="Render saved gym history deterministically")
    gym_history.add_argument("--goal-id", type=int, help="Goal containing imported gym logs")

    recommend = sub.add_parser("recommend-pullups", help="Generate and save a cautious pull-up next-step recommendation")
    recommend.add_argument("--goal-id", type=int, help="Goal containing imported gym logs")
    recommend.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:2b"))
    recommend.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))

    sub.add_parser("list-recommendations", help="List saved recommendations")

    feedback = sub.add_parser("recommendation-feedback", help="Accept, reject, or mark a recommendation modified")
    feedback.add_argument("recommendation_id", type=int)
    feedback.add_argument("status", choices=["accepted", "rejected", "modified"])
    feedback.add_argument("--note", default="")

    log = sub.add_parser("log-progress", help="Record a progress event")
    log.add_argument("goal_id", type=int)
    log.add_argument("value", type=float)
    log.add_argument("--note", default="")
    log.add_argument("--occurred-at", help="ISO timestamp; defaults to current UTC time")

    progress = sub.add_parser("progress", help="Calculate progress from saved events")
    progress.add_argument("goal_id", type=int)

    ask = sub.add_parser("ask", help="Ask Ollama using retrieved memory and calculated metrics")
    ask.add_argument("question")
    ask.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:2b"))
    ask.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    store = MemoryStore(args.db)

    try:
        if args.command == "create-goal":
            goal_id = store.create_goal(args.title, args.metric, args.target, args.unit, args.aggregation)
            print(f"Created goal #{goal_id}: {args.title}")
        elif args.command == "list-goals":
            goals = store.list_goals()
            print(json.dumps(goals, indent=2))
        elif args.command == "gym-history":
            rendered, evidence = render_gym_history(store, args.goal_id)
            print(rendered)
            print("\\nSTRUCTURED EVIDENCE")
            print(json.dumps(evidence, indent=2, ensure_ascii=False))
        elif args.command == "recommend-pullups":
            agent = MemoryAgent(store, OllamaClient(args.model, args.ollama_url))
            recommendation = agent.recommend_pullup_next_step(args.goal_id)
            print(f"Saved recommendation #{recommendation['id']} (status: {recommendation['status']}, confidence: {recommendation['confidence']})")
            print(recommendation["title"])
            print(recommendation["body"])
            print("\\nEvidence:")
            print(json.dumps(recommendation["evidence"], indent=2, ensure_ascii=False))
        elif args.command == "list-recommendations":
            print(json.dumps(store.list_recommendations(), indent=2, ensure_ascii=False))
        elif args.command == "recommendation-feedback":
            store.record_recommendation_feedback(args.recommendation_id, args.status, args.note)
            print(json.dumps(store.get_recommendation(args.recommendation_id), indent=2, ensure_ascii=False))
        elif args.command == "log-progress":
            event_id = store.record_progress(args.goal_id, args.value, args.note, args.occurred_at)
            print(f"Recorded event #{event_id} for goal #{args.goal_id}")
            print(json.dumps(calculate_goal_progress(store, args.goal_id), indent=2))
        elif args.command == "progress":
            print(json.dumps(calculate_goal_progress(store, args.goal_id), indent=2))
        elif args.command == "ask":
            agent = MemoryAgent(store, OllamaClient(args.model, args.ollama_url))
            answer, context = agent.ask(args.question)
            print("\nASSISTANT\n---------")
            print(answer)
            print("\nEVIDENCE CONTEXT (application-generated)\n----------------------------------------")
            print(json.dumps(context, indent=2, ensure_ascii=False))
    except (ValueError, OllamaError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
