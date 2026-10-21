# Multi-Head Intent Classifier Design

## Feature processing
The vectorizer creates word-level unigram through trigram TF-IDF features capped at 2,000 terms. Four dense values represent timestamp presence, a temporal-relation flag, character count, and word count. Dense signals use a stable two-dimensional row shape (1, 4), and combined matrices use `numpy.hstack` after validating matching row counts.

## Independent model heads
A shared feature matrix feeds separate Logistic Regression models: one predicts retrieval scope (`FRAME`, `CLIP`, `WHOLE_VIDEO`), while the second predicts reasoning tier (`L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`). Multi-route inference returns a second scope only when the top two probability values are within a configurable margin.

## Inference safety
Queries are normalized and empty strings are rejected before feature extraction. Labels are validated at fit time. The CLI trains a local model when the requested artifact does not exist. An untrained model instance delegates to the explicit rules baseline rather than making up confidence scores.

## Evaluation boundary
The bundled corpus is a small, manually constructed development/smoke corpus, not official Video-MME ground truth. It does not demonstrate generalization accuracy. No sub-5 ms latency guarantee is made. Measure latency and evaluate on a separate, reviewed labelled set before making performance claims.
