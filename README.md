# Problem 1.2 — Multi-Head User Intent Classifier

A CPU-oriented query-routing baseline with separate Logistic Regression heads for retrieval scope and reasoning tier, using word-level TF-IDF n-grams plus four dense query signals.

## Output labels
Retrieval scopes: `FRAME`, `CLIP`, `WHOLE_VIDEO`  
Reasoning tiers: `L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`

The classifier may return multiple candidate scopes when the top-two class probability margin is less than 0.15. A query has to be non-empty; an untrained instance uses an explicit rules fallback.

## Local setup (Python 3.11 recommended)
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
Open `http://127.0.0.1:8000/docs`. Endpoints: `GET /health` and `POST /route`.

## Docker
```bash
docker build -t user-intent-classifier:latest .
docker run --rm -p 8000:8000 user-intent-classifier:latest
```

## Evaluation integrity
The bundled corpus is a small development dataset, not official Video-MME ground truth. Accurate benchmark claims require reviewed FRAME/CLIP/WHOLE_VIDEO labels and an independent held-out evaluation. Sub-5 ms latency is not guaranteed; measure it in the target environment before reporting it.
