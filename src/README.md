# Source Code

| Module | Responsibility |
|---|---|
| `intent_classifier.py` | Query normalization, feature extraction, model fitting, inference and persistence |
| `train.py` | Small, explicitly labelled development corpus and local training entry point |
| `main.py` | CLI commands for training and prediction |
| `app.py` | FastAPI service, model lifecycle and routing endpoints |
| `benchmark.py` | Evaluation against a separately reviewed JSONL dataset |

## Design notes

- TF-IDF remains sparse through training and inference.
- Two Logistic Regression heads share features but predict separate outputs.
- The rules fallback is available to an untrained in-process classifier and emits no fabricated confidence.
- API startup requires a trained model artifact unless development startup training is explicitly enabled.
- Pickle artifacts must be treated as trusted local files; never load untrusted artifacts.
