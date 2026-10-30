"""Command-line interface for training and querying the intent classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train the development classifier")
    train_parser.add_argument("--model-path", type=Path, default=Path("models/pipeline.pkl"))
    predict_parser = subparsers.add_parser("predict", help="Route a single query")
    predict_parser.add_argument("--query", required=True)
    predict_parser.add_argument("--model-path", type=Path, default=Path("models/pipeline.pkl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_training_pipeline(str(args.model_path))
        return
    if not args.model_path.exists():
        raise SystemExit(
            f"Model not found at {args.model_path}. Run `python -m src.main train` first."
        )
    model = VideoRAGIntentClassifier.load(args.model_path)
    print(json.dumps(model.predict_route(args.query), indent=2))


if __name__ == "__main__":
    main()
