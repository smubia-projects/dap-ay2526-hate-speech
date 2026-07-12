"""Stateless FastAPI backend for the Guardrail Gauntlet web demo."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import ModelService
from rate_limit import check_rate_limit

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
model_service = ModelService()


def allowed_origins() -> list[str]:
    origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
    return [item for item in origins if item] or ["*"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(title="Singlish Hate Speech Guard API", version="1.0.0", lifespan=lifespan)
origins = allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=0.6, ge=0.2, le=0.95)


class PredictResponse(BaseModel):
    hate_probability: float
    verdict: Literal["allow", "review", "block"]
    threshold: float
    latency_ms: float
    model_repo_id: str


@app.get("/")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model_service.loaded,
        "model_repo_id": model_service.repo_id,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> dict[str, object]:
    check_rate_limit(request, "predict")
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail={"message": "The classifier is still warming up. Please try again."},
        )
    try:
        return model_service.predict(payload.text.strip(), payload.threshold)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "The classifier is unavailable. Please try again soon."},
        ) from exc
