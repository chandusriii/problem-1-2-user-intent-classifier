"""Evaluate routing with explicit top-1, candidate-recall and reasoning-tier metrics.

Expected JSONL schema per line:
{"query": "...", "scope": "FRAME", "tier": "L1_DIRECT"}

Only use an independently reviewed, held-out dataset. The bundled development
corpus must not be presented as benchmark or generalization evidence.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.intent_classifier import SCOPES, TIERS, VideoRAGIntentClassifier
from src.train import run_training_pipeline


def load_labelled_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
        clean = {key: row[key].strip() for key in required}
        if clean["scope"] not in SCOPES:
            raise ValueError(f"Line {line_number} has invalid scope label: {clean['scope']}")
        if clean["tier"] not in TIERS:
            raise ValueError(f"Line {line_number} has invalid tier label: {clean['tier']}")
        rows.append(clean)
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
    top1_scopes: list[str] = []
    tier_predictions: list[str] = []
    candidate_hits = 0
    multiple_candidate_count = 0
    latencies: list[float] = []

    for row in rows:
        start = time.perf_counter()
        result = model.predict_route(row["query"])
        latencies.append((time.perf_counter() - start) * 1000)
        candidates = result["assigned_scopes"]
        top1_scopes.append(candidates[0])
        tier_predictions.append(result["reasoning_tier"])
        candidate_hits += int(row["scope"] in candidates)
        multiple_candidate_count += int(len(candidates) > 1)

    ordered = sorted(latencies)
    # Nearest-rank p95 (rank = ceil(0.95 * n), converted to zero-based index).
    p95_index = max(0, min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "examples": len(rows),
        "scope_top1": classification_metrics(true_scopes, top1_scopes),
        "scope_candidate_recall": candidate_hits / len(rows),
        "multi_route_coverage": multiple_candidate_count / len(rows),
        "reasoning_tier": classification_metrics(true_tiers, tier_predictions),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": ordered[p95_index],
        },
        "evaluation_note": (
            "Metrics describe only this labelled dataset and runtime environment. "
            "Candidate recall is reported separately and is not top-1 accuracy. "
            "These are not Video-MME results unless labels and protocol are independently verified."
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
        parser.error(
            f"Model not found at {args.model}; train it first or opt into development training"
        )

    report = evaluate(model, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
