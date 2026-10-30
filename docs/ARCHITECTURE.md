# Architecture

## High-level flow

```text
Incoming query
  -> validation and normalization
  -> shared query feature extraction
       - word-level TF-IDF unigrams and bigrams (maximum 1,500 terms)
       - timestamp-presence indicator
       - temporal-language indicator
  -> independent prediction heads
       - retrieval scope: FRAME / CLIP / WHOLE_VIDEO
       - reasoning tier: L1_DIRECT / L2_RETRIEVAL / L3_GRAPH
  -> structured routing response
```

## Components

| Component | Responsibility |
|---|---|
| `intent_classifier.py` | Feature extraction, fitting, inference and versioned model persistence |
| `train.py` | Small, explicitly illustrative development corpus and training pipeline |
| `main.py` | Train and prediction CLI |
| `app.py` | FastAPI service and trained-model lifecycle |
| `benchmark.py` | Evaluation against separately reviewed JSONL labels |

The same normalization, vectorizer and structural-feature implementation is used in training and inference. TF-IDF remains sparse when the two binary structural indicators are appended.

## Deployment

The Docker build stage installs pinned dependencies and trains a development artifact. The runtime stage copies only the installed runtime, source and artifact into a slim Python image, runs as a non-root user, and exposes a health check. The API loads a trained artifact during application lifespan and refuses readiness when the artifact is absent, unless the explicit development-only startup-training switch is enabled.

## Evaluation boundary

The model is a text-only routing baseline. It does not itself inspect video frames, decode audio or reason over actual video content. The bundled hand-authored examples are engineering smoke-test fixtures, not Video-MME ground truth or an independent held-out dataset.
