"""FastAPI gateway over the multi-head intent classifier."""
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline

MODEL_PATH = Path("models/pipeline.pkl")
app = FastAPI(title="Video RAG User Intent Routing Service", version="2.0.0")
_classifier: VideoRAGIntentClassifier | None = None

def get_classifier() -> VideoRAGIntentClassifier:
    global _classifier
    if _classifier is None:
        if MODEL_PATH.exists():
            _classifier = VideoRAGIntentClassifier.load(str(MODEL_PATH))
        else:
            _classifier = run_training_pipeline(str(MODEL_PATH))
    return _classifier

class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)

class RouteResponse(BaseModel):
    query: str
    assigned_scopes: list[str]
    reasoning_tier: str
    confidence_distribution: dict[str, float]
    classifier: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/route", response_model=RouteResponse)
def process_retrieval_route(payload: RouteRequest) -> RouteResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    # If the model isn't ready, preserve truthful scope semantics via the rules fallback.
    result = get_classifier().predict_route(payload.query)
    if "assigned_scopes" not in result:
        raise HTTPException(status_code=500, detail="Classifier returned an invalid payload.")
    return RouteResponse(**result)
