"""CLI for the trained multi-head classifier with a safe rules fallback."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Video-related user query")
    parser.add_argument("--model-path", default="models/pipeline.pkl")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        run_training_pipeline(str(model_path))
    classifier = VideoRAGIntentClassifier.load(str(model_path))
    print(json.dumps(classifier.predict_route(args.query), indent=2))

if __name__ == "__main__":
    main()
