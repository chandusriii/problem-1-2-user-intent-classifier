# Architecture

## High-level flow

```text
Incoming query
  -> validation and normalization
  -> shared query feature extraction
       - TF-IDF word unigrams and bigrams
       - timestamp indicator
       - temporal-language indicator
       - character and word counts
  -> independent prediction heads
       - retrieval scope: FRAME / CLIP / WHOLE_VIDEO
       - reasoning tier: L1_DIRECT / L2_RETRIEVAL / L3_GRAPH
  -> structured response
```

## Components

| Component | Responsibility |
|---|---|
| `intent_classifier.py` | Feature extraction, fitting, inference and model persistence |
| `train.py` | Local development data and training pipeline |
| `main.py` | Train/predict CLI |
| `app.py` | HTTP service and trained model lifecycle |
| `benchmark.py` | Metrics on a separate labelled dataset |

The same feature transformer is used for fitting and inference. TF-IDF data remains sparse when concatenated with the four dense query features.

## Deployment

The Docker build stage trains the development model. The runtime stage copies the trained artifact into a slim Python image and launches the FastAPI application. The API refuses to start without a trained model unless explicit development startup training is enabled.
