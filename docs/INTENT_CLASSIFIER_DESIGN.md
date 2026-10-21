# Multi-Head Intent Classifier Design

## Feature processing
The vectorizer creates word-level unigram and bigram TF-IDF features capped at 1,500 terms. Four dense values represent timestamp presence, a temporal-relation flag, character count, and word count. Dense signals use a stable two-dimensional row shape (1, 4), and feature matrices use `numpy.hstack` after validating matching row counts. The single-query prediction path uses the same explicit alignment helper to avoid duplicated transformation code.

## Independent model heads
A shared feature matrix feeds separate Logistic Regression models: one predicts retrieval scope (`FRAME`, `CLIP`, `WHOLE_VIDEO`), while the second predicts reasoning tier (`L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`). Multi-route inference returns a second scope only when the top two probability values are within a configurable margin.

## API and container validation
The FastAPI service exposes `GET /health`, `POST /route`, and `POST /classify`; the latter two share the same request and response contract. GitHub Actions runs the unit/API tests, builds the multi-stage Docker image, starts the container, checks health, and sends live smoke-test requests to both inference endpoints.

## Inference safety
Queries are normalized and empty strings are rejected before feature extraction. Labels are validated at fit time. The CLI trains a local model when the requested artifact does not exist. An untrained model instance uses the explicit rules fallback rather than fabricating confidence scores.

## Evaluation boundary
The bundled corpus is a small, manually constructed development/smoke corpus, not official Video-MME ground truth. It does not demonstrate generalization accuracy. Small feature size and low latency are design goals, not measured guarantees. Measure serialized model size and request latency in the target environment and evaluate on a separate, reviewed labelled set before making performance claims.
