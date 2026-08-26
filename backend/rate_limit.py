"""Project adapter for the shared BIA FastAPI rate-limit engine."""

import os

from rate_limit_engine import RateLimiter

limiter = RateLimiter(
    project="dap-ay2526-hate-speech",
    buckets={"predict": int(os.getenv("RATE_LIMIT_PREDICT_MAX", "30"))},
    redis_url=os.getenv("UPSTASH_REDIS_REST_URL", os.getenv("UPSTASH_REDIS_URL", "")),
    redis_token=os.getenv(
        "UPSTASH_REDIS_REST_TOKEN", os.getenv("UPSTASH_REDIS_TOKEN", "")
    ),
    default_window=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
)


def check_rate_limit(request, bucket: str) -> None:
    limiter.enforce(request, bucket)
