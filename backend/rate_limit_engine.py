"""Centralised rate-limit engine for FastAPI showcase projects."""

from __future__ import annotations

import time
from typing import NamedTuple, Optional

ENGINE_VERSION = "1.3.0"


class RateLimitStatus(NamedTuple):
    status: str
    max: Optional[int]


def client_ip(request) -> str:
    """Return the real client IP when running behind HF Spaces' proxy."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(
        self,
        project: str,
        buckets: dict[str, int],
        *,
        redis_url: str = "",
        redis_token: str = "",
        default_window: int = 432000,
        cache_ttl: int = 90,
    ) -> None:
        self.project = project
        self.buckets = dict(buckets)
        self._redis_url = redis_url
        self._redis_token = redis_token
        self.default_window = default_window
        self.cache_ttl = cache_ttl
        self._redis = None
        self._cache = {
            "t": 0.0,
            "salt": "",
            "killed": False,
            "limits": dict(buckets),
            "window": default_window,
        }

    def _get_redis(self):
        if self._redis is None:
            if not self._redis_url or not self._redis_token:
                return None
            from upstash_redis import Redis

            self._redis = Redis(url=self._redis_url, token=self._redis_token)
        return self._redis

    def _refresh(self, redis) -> None:
        now = time.time()
        if now - self._cache["t"] < self.cache_ttl:
            return
        try:
            keys = [
                "config:kill_all",
                "config:event_mode",
                "config:ratelimit_salt",
                f"config:{self.project}:mode_override",
                f"config:{self.project}:enabled",
            ]
            for mode in ("default", "demo"):
                keys.append(f"config:{self.project}:{mode}:window")
                keys.extend(
                    f"config:{self.project}:{mode}:{bucket}" for bucket in self.buckets
                )
            data = dict(zip(keys, redis.mget(*keys)))
            killed = (
                str(data["config:kill_all"]) == "1"
                or str(data[f"config:{self.project}:enabled"]) == "0"
            )
            mode = (
                data[f"config:{self.project}:mode_override"]
                or data["config:event_mode"]
                or "default"
            )
            if mode not in ("default", "demo"):
                mode = "default"
            salt = data["config:ratelimit_salt"] or ""
            window_value = data[f"config:{self.project}:{mode}:window"]
            window = int(window_value) if window_value is not None else self.default_window
            limits = {}
            for bucket, env_default in self.buckets.items():
                value = data[f"config:{self.project}:{mode}:{bucket}"]
                limits[bucket] = int(value) if value is not None else env_default
            self._cache.update(
                t=now,
                salt=salt,
                killed=killed,
                limits=limits,
                window=window,
            )
        except Exception:
            self._cache.update(
                t=now,
                killed=False,
                limits=dict(self.buckets),
                window=self.default_window,
            )

    def status(self, ip: str, bucket: str) -> RateLimitStatus:
        redis = self._get_redis()
        if redis is None:
            return RateLimitStatus("ok", None)
        self._refresh(redis)
        if self._cache["killed"]:
            return RateLimitStatus("killed", None)

        max_requests = self._cache["limits"].get(bucket, self.buckets.get(bucket, 0))
        if max_requests <= 0:
            return RateLimitStatus("ok", None)

        key = f"ratelimit:{self._cache['salt']}:{self.project}:{bucket}:{ip}"
        try:
            count = redis.incr(key)
            if count == 1:
                redis.expire(key, self._cache["window"])
        except Exception:
            return RateLimitStatus("ok", None)
        if count > max_requests:
            return RateLimitStatus("limited", max_requests)
        return RateLimitStatus("ok", None)

    def enforce(self, request, bucket: str) -> None:
        state, value = self.status(client_ip(request), bucket)
        if state == "killed":
            from fastapi import HTTPException

            raise HTTPException(
                status_code=503,
                detail={"message": "This demo is temporarily paused. Check back soon."},
            )
        if state == "limited":
            from fastapi import HTTPException

            raise HTTPException(
                status_code=429,
                detail={"message": "Rate limit exceeded.", "queries_used": value},
            )
