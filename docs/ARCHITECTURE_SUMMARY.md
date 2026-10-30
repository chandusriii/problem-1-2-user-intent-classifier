# Problem 1.2 — Architecture Summary

This document summarizes the current implementation. The companion slide deck, if retained in the repository, is a presentation artifact and is not a source of benchmark evidence.

## 1. Routing dimensions

The classifier independently predicts retrieval scope (`FRAME`, `CLIP`, `WHOLE_VIDEO`) and reasoning tier (`L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`). It is a text-based routing baseline and does not itself inspect video frames or decode audio.

## 2. Feature engineering

The model combines word-level TF-IDF unigrams/bigrams (maximum 1,500 terms) with two bounded binary structural signals: timestamp presence and temporal-language presence. Two Logistic Regression heads share the sparse feature representation. A second candidate scope may be emitted when the top-two class probabilities are within the configured margin.

## 3. Runtime and guardrails

The FastAPI application loads a trained, versioned model artifact during startup and fails startup when the artifact is missing unless explicit development-only startup training is enabled. The Docker image uses a multi-stage build and a non-root runtime user. The unified GitHub Actions workflow trains the development artifact, exercises the CLI, runs tests, builds the image, checks container health, and exercises both inference endpoints.

## Evaluation boundary

The bundled training corpus and `data/intent_eval.jsonl` smoke sample are illustrative engineering data, not a held-out evaluation set or official Video-MME ground truth. The benchmark now reports top-1 scope accuracy separately from candidate recall, multi-route coverage, and reasoning-tier metrics. No generalization or latency claim should be made without a larger independently reviewed dataset and reproducible measurements.
