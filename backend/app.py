"""Stateless FastAPI backend for the Beat the Guard web demo.

Scores one message with two guards (SG-Tuned SingBERT + a standard toxicity foil)
and returns both, so the frontend can show them side by side.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import GuardService
from rate_limit import check_rate_limit

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
guard_service = GuardService()


def allowed_origins() -> list[str]:
    origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
    return [item for item in origins if item] or ["*"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    guard_service.load()
    yield


app = FastAPI(title="Beat the Guard API", version="2.0.0", lifespan=lifespan)
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


class GuardResult(BaseModel):
    key: str
    name: str
    metric: str
    probability: float
    verdict: Literal["allow", "review", "block"]
    latency_ms: float
    model_repo_id: str


class PredictResponse(BaseModel):
    text: str
    threshold: float
    guards: list[GuardResult]


@app.get("/")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": guard_service.loaded,
        "guards": guard_service.status(),
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> dict[str, object]:
    check_rate_limit(request, "predict")
    if not guard_service.loaded:
        raise HTTPException(
            status_code=503,
            detail={"message": "The guards are still warming up. Please try again."},
        )
    try:
        return guard_service.predict(payload.text.strip(), payload.threshold)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "The guards are unavailable. Please try again soon."},
        ) from exc
