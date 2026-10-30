"""FastAPI gateway for multi-head video-query routing."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "pipeline.pkl"
classifier: VideoRAGIntentClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load a model artifact; never silently train during production startup."""
    global classifier
    model_path = Path(os.getenv("INTENT_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    try:
        if model_path.exists():
            classifier = VideoRAGIntentClassifier.load(model_path)
        elif os.getenv("INTENT_ALLOW_STARTUP_TRAINING", "false").lower() == "true":
            logger.warning("Missing model; explicit development startup training is enabled.")
            classifier = run_training_pipeline(str(model_path))
        else:
            raise RuntimeError(
                f"Model artifact not found at {model_path}. Train with `python -m src.main train` "
                "or explicitly enable INTENT_ALLOW_STARTUP_TRAINING for development."
            )
        if not classifier.is_trained:
            raise RuntimeError("Loaded classifier artifact is not trained.")
        logger.info("Intent classifier loaded from %s", model_path)
        yield
    except Exception:
        logger.exception("Unable to initialize the intent classifier.")
        raise
    finally:
        classifier = None


app = FastAPI(
    title="Multi-Head Video RAG Intent Router",
    description="Routes video-related queries by retrieval scope and reasoning tier.",
    version="3.0.0",
)


class RoutingRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=5000,
        description="Video-related user query.",
        examples=["What color is the car passing by at 00:15?"],
    )


class RoutingResponse(BaseModel):
    query: str
    assigned_scopes: list[str]
    reasoning_tier: str
    confidence_distribution: dict[str, float]
    classifier: str


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Report readiness only when an actual trained model is loaded."""
    if classifier is not None and classifier.is_trained:
        return {"status": "healthy", "engine_state": "online"}
    raise HTTPException(status_code=503, detail="Inference engine is not ready.")


@app.post("/route", response_model=RoutingResponse)
@app.post("/classify", response_model=RoutingResponse)
def route_query(payload: RoutingRequest) -> RoutingResponse:
    """Predict retrieval scopes and reasoning tier for a query."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    if classifier is None or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")
    try:
        result: dict[str, Any] = classifier.predict_route(query)
        return RoutingResponse(**result)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail="Inference pipeline failed.") from exc
