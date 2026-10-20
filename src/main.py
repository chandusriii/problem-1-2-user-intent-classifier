"""Command-line entry point for the lightweight user-intent classifier."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import classify_batch, classify_query

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="One video-related query")
    parser.add_argument("--queries-file", help="Text file with one query per line")
    parser.add_argument("--help-examples", action="store_true")
    args = parser.parse_args()
    if args.help_examples:
        print('Examples:\n  python src/main.py --query "What color is the car at 00:15?"\n  python src/main.py --query "Summarize the entire video."')
        return
    if args.query and args.queries_file:
        parser.error("choose either --query or --queries-file")
    if args.query:
        print(json.dumps(classify_query(args.query), indent=2))
    elif args.queries_file:
        path = Path(args.queries_file)
        if not path.is_file():
            parser.error(f"queries file not found: {path}")
        queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(classify_batch(queries), indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
