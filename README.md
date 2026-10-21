# Problem 1.2 — User Intent Classifier

A CPU-oriented multi-head query-routing baseline. It predicts retrieval scope and reasoning tier independently using TF-IDF features, four dense structural signals, and two Logistic Regression heads.

## Outputs
- Retrieval scopes: `FRAME`, `CLIP`, `WHOLE_VIDEO`
- Reasoning tiers: `L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`

When the top two scope probabilities are close (margin below 0.15), inference returns both candidates. An untrained model uses an explicit rules fallback rather than a fabricated confidence score.

## Run locally (Python 3.11 recommended)
```bash
python -m pip install -r requirements.txt
python src/train.py
python src/main.py --query "What color was the car at 00:15?"
python -m unittest discover -s tests -v
```

## API
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs`. The service offers `GET /health` and `POST /route`.

## Docker
```bash
docker build -t user-intent-classifier:latest .
docker run --rm -p 8000:8000 user-intent-classifier:latest
```

## Evaluation honesty
The bundled development corpus is small and manually constructed to exercise the pipeline. It is not official Video-MME ground truth. The provided Video-MME questions do not automatically label retrieval scope, so real accuracy claims require reviewed FRAME/CLIP/WHOLE_VIDEO annotations and a held-out evaluation. Sub-5 ms latency is a target, not a measured guarantee; benchmark latency on the actual environment before claiming it.
