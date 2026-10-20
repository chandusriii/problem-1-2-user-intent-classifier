"""Evaluate scope predictions on a labelled JSONL query dataset."""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import classify_query

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="JSONL rows with query and scope")
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No labelled examples found.")
    y_true, y_pred, times = [], [], []
    for row in rows:
        start = time.perf_counter()
        pred = classify_query(row["query"])
        times.append(time.perf_counter() - start)
        y_true.append(row["scope"])
        y_pred.append(pred["scope"])
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    result = {
        "examples": len(rows),
        "scope_accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(rows),
        "scope_macro_f1": sum(v["f1"] for v in per_class.values()) / len(per_class),
        "per_class": per_class,
        "mean_latency_ms": 1000 * sum(times) / len(times),
        "p95_latency_ms": 1000 * sorted(times)[min(len(times)-1, int(0.95*len(times)))],
        "warning": "Metrics only describe these labelled examples; do not call them VideoMME benchmark results unless the VideoMME sample has been separately scope-labelled and evaluated."
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
