# Limitations and Evaluation Boundaries

## Current model

This project is an interpretable lexical routing baseline built with TF-IDF and two Logistic Regression heads. It predicts a retrieval scope and a reasoning tier from query text. It does not inspect video frames, decode audio, or understand the video itself.

## Training data

The bundled corpus is small and manually authored. It is useful for exercising the software pipeline, but it is not a representative production dataset or official Video-MME ground truth. It should not be used as a held-out evaluation set for the same model trained on those examples.

## Model confidence

The scope head exposes Logistic Regression probabilities. These values are not guaranteed to be calibrated. The optional second scope is selected using a configurable probability-margin heuristic, not a calibrated uncertainty estimate.

## Operational claims

Latency, memory use and model size depend on runtime, hardware, dependencies and serving configuration. Publish measurements only after reproducible measurement in a named environment.

## Safe next steps

1. Create an annotation guide for scope and reasoning labels.
2. Collect a larger and more diverse query set.
3. Use train/validation/test splits that prevent near-duplicate leakage.
4. Report per-class metrics and error analysis on the held-out test set.
5. Calibrate probabilities before interpreting them as confidence.
6. Compare stronger text encoders against this baseline using the same protocol.
