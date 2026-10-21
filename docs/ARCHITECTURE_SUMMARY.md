# Problem 1.2 — Three-Slide Architecture Summary

This repository's companion slide deck is available as a downloadable artifact from the project conversation: `problem_1_2_architecture_summary.pptx`.

## Slide 1 — Architectural intent
The router separates **retrieval scope** (`FRAME`, `CLIP`, `WHOLE_VIDEO`) from **reasoning tier** (`L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`). A small CPU-oriented classifier routes queries before downstream retrieval.

## Slide 2 — Feature engineering
The pipeline combines word-level TF-IDF unigram/bigram features (up to 1,500 terms) with four dense signals: timestamp presence, temporal cue, character count, and word count. Two independent Logistic Regression heads predict scope and reasoning tier. When the top-two scope probabilities are close (margin below 0.15), a second scope may be returned.

## Slide 3 — Production guardrails
GitHub Actions runs Python tests, builds the multi-stage Docker image, starts the container, checks `/health`, and sends live requests to `POST /route` and `POST /classify`.

## Evidence and limitations
The automated validation workflow has passed its code-test, Docker-build, container health, and API smoke-test stages on the checked run. The packaged training examples are a small development corpus, not official Video-MME labels. The presentation intentionally does not claim sub-1 ms latency, sub-5 ms latency, or a <5 MB memory footprint as measured facts; those require separate measurements.
