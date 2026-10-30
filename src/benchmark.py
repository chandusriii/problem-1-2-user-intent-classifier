"""Evaluate both prediction heads on an independently reviewed JSONL dataset.

Expected schema per line:
{"query": "...", "scope": "FRAME", "tier": "L1_DIRECT"}

Do not reuse development training examples as held-out evaluation data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline


def load_labelled_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object")
        required = {"query", "scope", "tier"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Line {line_number} is missing keys: {sorted(missing)}")
        if not all(isinstance(row[key], str) and row[key].strip() for key in required):
            raise ValueError(f"Line {line_number} contains an empty or non-string field")
        rows.append({key: row[key].strip() for key in required})
    if not rows:
        raise ValueError("No labelled examples found")
    return rows


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def evaluate(model: VideoRAGIntentClassifier, rows: list[dict[str, str]]) -> dict[str, Any]:
    true_scopes = [row["scope"] for row in rows]
    true_tiers = [row["tier"] for row in rows]
    predicted_scopes, predicted_tiers, latencies = [], [], []

    for row in rows:
        start = time.perf_counter()
        result = model.predict_route(row["query"])
        latencies.append((time.perf_counter() - start) * 1000)
        predicted_scopes.append(
            row["scope"] if row["scope"] in result["assigned_scopes"] else result["assigned_scopes"][0]
        )
        predicted_tiers.append(result["reasoning_tier"])

    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "examples": len(rows),
        "scope": classification_metrics(true_scopes, predicted_scopes),
        "reasoning_tier": classification_metrics(true_tiers, predicted_tiers),
        "latency_ms": {"mean": sum(latencies) / len(latencies), "p95": ordered[p95_index]},
        "evaluation_note": (
            "Metrics describe only this labelled dataset and runtime environment. "
            "They are not Video-MME results unless labels and protocol are independently verified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Reviewed JSONL evaluation labels")
    parser.add_argument("--model", type=Path, default=Path("models/pipeline.pkl"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--train-development-model", action="store_true")
    args = parser.parse_args()
    rows = load_labelled_rows(args.labels)

    if args.model.exists():
        model = VideoRAGIntentClassifier.load(args.model)
    elif args.train_development_model:
        model = run_training_pipeline(str(args.model))
    else:
        parser.error(f"Model not found at {args.model}; train it first or opt into development training")

    report = evaluate(model, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
