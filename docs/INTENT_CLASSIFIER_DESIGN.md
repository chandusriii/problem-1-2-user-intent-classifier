# Multi-Head Intent Classifier Design

## Feature processing

The model uses word-level unigram and bigram TF-IDF features capped at 1,500 terms, plus two binary structural indicators: timestamp presence and temporal-language presence. Character and token counts were removed because their unscaled magnitudes could dominate the normalized TF-IDF feature space. The TF-IDF matrix remains sparse when structural signals are appended.

The exact same normalization, vectorizer and structural-feature path is used for both training and inference.

## Independent model heads

A shared feature matrix feeds separate Logistic Regression models. The scope head predicts `FRAME`, `CLIP` or `WHOLE_VIDEO`; the reasoning head predicts `L1_DIRECT`, `L2_RETRIEVAL` or `L3_GRAPH`. The scope head may return a second candidate when the two highest class probabilities are within a configurable margin. Candidate recall is reported separately from top-1 accuracy.

## Model artifact

Artifacts are Python pickle files and must only be loaded from trusted sources. The model stores an explicit format version and rejects older or incompatible artifacts. Retrain after changing model code or dependencies; pin the Python and scikit-learn runtime used to train and serve the artifact.

## API and container validation

The FastAPI service loads its trained model through its lifespan handler. It exposes `GET /health`, `POST /route` and `POST /classify`; the two POST routes share a response contract. Startup fails when the model artifact is absent unless the explicit development-only training switch is enabled. CI exercises API startup, both inference endpoints and the Docker runtime.

## Inference safety

Queries are normalized and empty strings are rejected before feature extraction. Training rejects invalid labels, mismatched arrays and single-class heads. The untrained in-memory classifier has a rules fallback that reports an empty probability distribution rather than fabricated confidence scores.

## Evaluation boundary

The included training corpus is small and manually authored. The bundled JSONL evaluation example is a schema/smoke sample, not an independently reviewed held-out benchmark. Neither should be used to make Video-MME accuracy claims or claims of generalization.
