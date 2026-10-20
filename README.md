# Problem 1.2 — User Intent Classifier

This repository implements an initial CPU-only query classifier for the AIntropy task. Its required primary labels are `FRAME`, `CLIP`, and `WHOLE_VIDEO`; an additional AdaVideoRAG-inspired difficulty tier (`L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`) is kept separate from scope.

## Run

```bash
python -m pip install -r requirements.txt
python src/main.py --help
python src/main.py --query "What color is the car at 00:15?"
python src/main.py --query "When does the presenter explain neural networks?"
python src/main.py --query "Summarize the entire video."
python src/benchmark.py --labels data/intent_eval.jsonl --output results/benchmark.json
python -m unittest discover -s tests -v
```

## Components
- `src/intent_classifier.py`: query feature engineering, scope classification, and route recommendation.
- `src/main.py`: single-query and batch CLI.
- `src/benchmark.py`: accuracy, macro-F1, per-class precision/recall, and latency on an explicitly labelled JSONL file.
- `docs/INTENT_CLASSIFIER_DESIGN.md`: design, limitations, VideoMME annotation/evaluation plan.
- `run.yaml`: exact grader setup, run, and test commands.

## Evaluation note
`data/intent_eval.jsonl` is a small manually-labelled smoke-test set for exercising the benchmark code. It is **not** the required VideoMME 10% benchmark. The provided VideoMME questions do not automatically tell us whether their retrieval scope is frame, clip, or whole video. A real benchmark result requires downloading the assigned sample, applying a documented scope annotation policy to its queries, reviewing the labels, and running evaluation.

The baseline is deliberately transparent and may fail on paraphrases or compound queries. Do not claim perfect routing or production accuracy based on the smoke set.
