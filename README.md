<div align="center">

# Multi-Head Video RAG Intent Classifier

**A lightweight query router for retrieval scope and reasoning tier.**

[![CI](https://github.com/chandusriii/problem-1-2-user-intent-classifier/actions/workflows/verify.yml/badge.svg)](https://github.com/chandusriii/problem-1-2-user-intent-classifier/actions/workflows/verify.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-see%20LICENSE-lightgrey)](LICENSE)

</div>

---

## Overview

This repository implements a CPU-oriented routing baseline for video question answering systems. It classifies an incoming query along two independent dimensions:

- **Retrieval scope:** which portion of a video is likely to be relevant.
- **Reasoning tier:** how much retrieval or temporal reasoning may be required.

Independent Logistic Regression heads operate over a shared feature representation. The repository includes a Python API, CLI, Docker packaging, tests and continuous integration.

> **Evaluation boundary:** bundled training examples are a small, manually constructed development corpus. They are not official Video-MME labels and do not establish generalization. No latency, memory or benchmark accuracy claim is made without independent measurement.

## Prediction contract

| Dimension | Label | Interpretation |
|---|---|---|
| Retrieval scope | `FRAME` | Specific visual moment or frame |
| | `CLIP` | Bounded segment, action or utterance |
| | `WHOLE_VIDEO` | Global summary, theme or end-to-end timeline |
| Reasoning tier | `L1_DIRECT` | Direct answer from a narrowly scoped observation |
| | `L2_RETRIEVAL` | Locate and inspect relevant video evidence |
| | `L3_GRAPH` | Multi-step, temporal or relationship-oriented reasoning |

When the two highest scope probabilities are close, the model may return two candidate scopes. An untrained classifier instance can use a rules fallback; the API requires a trained artifact by default.

## Architecture

```text
User query
    |
    v
Normalization
    |
    v
Shared features: word TF-IDF (1-2 grams) + timestamp + temporal cue
                 + character count + word count
    |
    +-----------------------------+
    |                             |
    v                             v
Scope Logistic Regression   Reasoning Logistic Regression
    |                             |
    v                             v
FRAME / CLIP / WHOLE_VIDEO   L1 / L2 / L3
    |                             |
    +--------------+--------------+
                   v
           Structured route result
```

### Feature design

- Word-level TF-IDF unigrams and bigrams, capped at 1,500 features.
- Two binary structural signals: timestamp presence and temporal-language presence.
- Sparse feature matrices are preserved through model fitting and inference; structural indicators are bounded to avoid overwhelming TF-IDF features.
- Both heads use deterministic Logistic Regression configuration.

## Repository layout

```text
.
├── .github/workflows/       # Automated validation
├── data/                    # Development/evaluation data and notes
├── docs/                    # Architecture and design docs
├── src/
│   ├── app.py               # FastAPI service
│   ├── benchmark.py         # Labelled-set evaluation
│   ├── intent_classifier.py # Feature extraction and model
│   ├── main.py              # CLI
│   └── train.py             # Development corpus and training
├── tests/                   # Unit and API tests
├── Dockerfile
├── requirements.txt
└── run.yaml
```

## Quick start

Python 3.11 is the supported CI runtime.

**Windows PowerShell**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install packages and train:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.main train
```

Run predictions:

```bash
python -m src.main predict --query "What color was the car at 00:15?"
python -m src.main predict --query "Summarize the entire video."
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## API

Start after training:

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Route a query:

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query":"What color was the car at 00:15?"}'
```

Example response schema (illustrative only; probabilities are not measured project output):

```json
{
  "query": "What color was the car at 00:15?",
  "assigned_scopes": ["FRAME"],
  "reasoning_tier": "L1_DIRECT",
  "confidence_distribution": {"FRAME": 0.72, "CLIP": 0.19, "WHOLE_VIDEO": 0.09},
  "classifier": "tfidf_logistic_regression_multi_head"
}
```

Both `POST /route` and `POST /classify` are supported. By default, the API does not train at startup. Set `INTENT_MODEL_PATH` to load another artifact path. For development only, `INTENT_ALLOW_STARTUP_TRAINING=true` explicitly enables training when the artifact is absent.

## Docker

```bash
docker build -t video-intent-router .
docker run --rm -p 8000:8000 video-intent-router
```

The multi-stage image trains the development artifact during build and copies it into a slim runtime image. The container health check uses `/health`.

## Evaluation

Use an independently reviewed JSONL dataset:

```json
{"query":"What color is the car at 00:15?","scope":"FRAME","tier":"L1_DIRECT"}
{"query":"Summarize the entire recording.","scope":"WHOLE_VIDEO","tier":"L2_RETRIEVAL"}
```

Run:

```bash
python -m src.benchmark --labels path/to/reviewed_eval.jsonl \
  --model models/pipeline.pkl \
  --output results/benchmark.json
```

The evaluator reports scope top-1 accuracy and macro F1 separately from candidate recall and multi-route coverage. Reasoning-tier metrics are evaluated independently. Candidate recall is not conventional top-1 accuracy.

The bundled JSONL file illustrates the required schema; it is not an independently reviewed held-out benchmark. Do not use the training corpus or this illustrative sample as evidence of generalization. Publish dataset provenance, annotation rules, split strategy, package versions, artifact version and runtime environment with any reported result.

## Engineering safeguards

- Rejects non-string, blank and oversized queries.
- Validates labels and training input lengths.
- Preserves sparse TF-IDF representations.
- Separates training, CLI, API and evaluation responsibilities.
- Keeps generated model artifacts out of source control.
- Runs unit tests, API tests and Docker integration checks.
- Requires a trained artifact at API startup by default.
- Never fabricates confidence for rules-based fallback.

## Limitations and roadmap

This is a baseline router, not a complete video understanding system. Lexical query features cannot fully infer visual evidence, temporal complexity or user intent. The small hand-authored development corpus is suitable for engineering smoke tests only.

Next steps: collect a larger reviewed corpus, define annotation guidance, measure per-class performance on held-out data, calibrate confidence and compare compact sentence-embedding models under the same evaluation protocol.

## Documentation

- [Evaluation protocol](docs/EVALUATION.md)
- [Limitations](docs/LIMITATIONS.md)
- [Classifier design](docs/INTENT_CLASSIFIER_DESIGN.md)
- [Architecture summary](docs/ARCHITECTURE_SUMMARY.md)
- [Source module guide](src/README.md)

## License

See [LICENSE](LICENSE).
