# Problem 1.2 — Engineering and Evaluation Summary

## Problem

Video question-answering systems need an early routing step that decides what span of a recording to retrieve and what level of reasoning is likely to be needed. This repository implements a lightweight text-only baseline for that routing decision.

## Approach

A shared word-level TF-IDF representation (unigrams and bigrams, capped at 1,500 terms) is combined with two bounded binary signals: timestamp presence and temporal-language presence. Two independent Logistic Regression heads predict:

- Retrieval scope: `FRAME`, `CLIP`, or `WHOLE_VIDEO`.
- Reasoning tier: `L1_DIRECT`, `L2_RETRIEVAL`, or `L3_GRAPH`.

The TF-IDF matrix stays sparse through feature assembly. At inference, the scope head can return a second candidate when the top two class probabilities are within a configurable margin.

## Engineering

The implementation is separated into classifier, training, CLI, API, and benchmark modules. The FastAPI service loads a versioned model artifact through its lifespan and refuses to become ready when the model is absent. The Docker image uses a multi-stage build and runs as a non-root user. GitHub Actions installs pinned dependencies, trains the development artifact, smoke-tests CLI routes, runs unit/API tests, validates the JSONL contract, builds Docker, and exercises health plus both inference endpoints.

## Evaluation integrity

The bundled corpus and JSONL sample are small manually written engineering fixtures. They are not official Video-MME labels, not an independent held-out set, and not evidence of generalization. The benchmark therefore reports top-1 scope metrics separately from multi-route candidate recall and coverage, plus reasoning-tier metrics and environment-dependent latency. No accuracy or latency claim is made here.

A meaningful model-quality evaluation still requires a larger independently annotated dataset, a written labelling guide, leakage-safe train/validation/test splits, per-class error analysis, calibration checks, and reproducible measurements on a named runtime/hardware configuration.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m src.main train
python -m src.main predict --query "What color is the car at 00:15?"
python -m unittest discover -s tests -v
```

Start the API after training with `uvicorn src.app:app --host 127.0.0.1 --port 8000`. See the root README for Docker, request examples, and benchmark usage.
