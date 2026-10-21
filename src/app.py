"""FastAPI gateway around the same stateless classifier as the CLI."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.intent_classifier import classify_query

app = FastAPI(
    title="Video RAG User Intent Routing Service",
    description="Routes video questions by retrieval scope and reasoning difficulty.",
    version="1.0.0",
)

class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)

class RouteResponse(BaseModel):
    query: str
    assigned_scope: str
    reasoning_tier: str
    retrieval_strategy: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/route", response_model=RouteResponse)
def process_retrieval_route(payload: RouteRequest) -> RouteResponse:
    clean_text = payload.query.strip()
    if not clean_text:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    result = classify_query(clean_text)
    return RouteResponse(
        query=result["query"],
        assigned_scope=result["scope"],
        reasoning_tier=result["level"],
        retrieval_strategy=result["retrieval_strategy"],
    )
