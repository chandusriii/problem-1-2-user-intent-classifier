"""FastAPI gateway over the multi-head intent classifier."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.intent_classifier import VideoRAGIntentClassifier
from src.train import run_training_pipeline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline.pkl"
classifier: VideoRAGIntentClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model before serving traffic; train only as a deployment fallback."""
    global classifier
    try:
        if MODEL_PATH.exists():
            classifier = VideoRAGIntentClassifier.load(str(MODEL_PATH))
        else:
            logger.warning("Trained model asset missing; training the development model at startup.")
            classifier = run_training_pipeline(str(MODEL_PATH))
        if not classifier.is_trained:
            raise RuntimeError("Loaded classifier is not trained.")
        logger.info("Intent classifier loaded and ready.")
        yield
    except Exception:
        logger.exception("Unable to initialize the intent classifier.")
        raise
    finally:
        classifier = None


app = FastAPI(
    title="Multi-Head Video RAG Intent Routing Gateway",
    description="Routes video-related queries by retrieval scope and reasoning tier.",
    version="2.1.0",
    lifespan=lifespan,
)


class RoutingRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw video-related user query.",
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
    """Report process readiness based on whether a trained model is loaded."""
    if classifier is not None and classifier.is_trained:
        return {"status": "healthy", "engine_state": "online"}
    raise HTTPException(status_code=503, detail="Inference engine is not ready.")


@app.post("/route", response_model=RoutingResponse, status_code=status.HTTP_200_OK)
def route_query(payload: RoutingRequest) -> RoutingResponse:
    """Predict retrieval scope and reasoning tier for one query."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be blank.",
        )
    if classifier is None or not classifier.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not ready.",
        )
    try:
        result: dict[str, Any] = classifier.predict_route(query)
        return RoutingResponse(
            query=result["query"],
            assigned_scopes=result["assigned_scopes"],
            reasoning_tier=result["reasoning_tier"],
            confidence_distribution=result["confidence_distribution"],
            classifier=result["classifier"],
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference pipeline failed.",
        ) from exc
