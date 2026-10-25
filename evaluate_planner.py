#!/usr/bin/env python3
"""Evaluate planner interpretation against a small labelled natural-language test set.

Run from the project root after copying this file and planner_eval.json there:
  python evaluate_planner.py --model qwen3.5:2b
  python evaluate_planner.py --model qwen3.5:4b

This evaluates LLM planning only, not database retrieval or final answer correctness.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from memory_engine.stage1_chat import PLANNER_SYSTEM_PROMPT

FIELDS = ("intent", "query_type", "exercise_name", "session_label", "session_date", "load_type", "include_dates")


def normalize_exercise(value: Any) -> Any:
    if value is None:
        return None
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    if len(key) > 4 and key.endswith("s"):
        key = key[:-1]
    aliases = {"pullup": "pullup", "pushup": "pushup"}
    return aliases.get(key, key)


def ask_model(model: str, message: str, url: str, timeout: int) -> tuple[dict[str, Any] | None, float, str | None]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": "CURRENT_DATE: " + date.today().isoformat() + "\nUSER_MESSAGE:\n" + message},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_predict": 256},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed = time.perf_counter() - started
        content = payload.get("message", {}).get("content", "")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None, elapsed, "Model output was JSON but not an object"
        return parsed, elapsed, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def equal(field: str, actual: Any, expected: Any) -> bool:
    if field == "exercise_name":
        return normalize_exercise(actual) == normalize_exercise(expected)
    if field in {"session_label"}:
        return (str(actual).casefold().strip() if actual is not None else None) == (str(expected).casefold().strip() if expected is not None else None)
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dataset", default="planner_eval.json")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 means all)")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text())
    if args.limit:
        cases = cases[:args.limit]
    totals = {field: 0 for field in FIELDS}
    correct = {field: 0 for field in FIELDS}
    case_results = []
    print(f"Model: {args.model} | Cases: {len(cases)} | Endpoint: {args.url}")
    for index, case in enumerate(cases, 1):
        actual, elapsed, error = ask_model(args.model, case["message"], args.url, args.timeout)
        expected = case["expected"]
        mismatches = []
        if error:
            mismatches.append("ERROR=" + error)
        else:
            for field in FIELDS:
                if field not in expected:
                    continue
                totals[field] += 1
                got = actual.get(field)
                want = expected[field]
                if equal(field, got, want):
                    correct[field] += 1
                else:
                    mismatches.append(f"{field}: expected {want!r}, got {got!r}")
        status = "PASS" if not mismatches else "FAIL"
        print(f"[{index:02d}/{len(cases):02d}] {status} {case['id']} ({elapsed:.2f}s)")
        for mismatch in mismatches:
            print("   " + mismatch)
        case_results.append({"id": case["id"], "message": case["message"], "expected": expected,
                             "actual": actual, "seconds": round(elapsed, 3), "errors": mismatches})

    print("\nField accuracy:")
    for field in FIELDS:
        if totals[field]:
            print(f"  {field:16s} {correct[field]}/{totals[field]} = {correct[field] / totals[field]:.1%}")
    fully_correct = sum(not item["errors"] for item in case_results)
    print(f"\nExact labelled cases: {fully_correct}/{len(case_results)} = {fully_correct / max(1, len(case_results)):.1%}")
    out = Path(f"planner_eval_results_{re.sub(r'[^a-zA-Z0-9_.-]', '_', args.model)}.json")
    out.write_text(json.dumps({"model": args.model, "case_count": len(case_results), "field_correct": correct,
                               "field_total": totals, "results": case_results}, indent=2))
    print(f"Detailed results written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
