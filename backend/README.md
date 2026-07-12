---
title: Singlish Guardrail API
emoji: "🛡️"
colorFrom: violet
colorTo: indigo
sdk: docker
pinned: false
---

# Singlish Guardrail API

CPU-only FastAPI inference for the Guardrail Gauntlet. The Space downloads the
public SingBERT model from Hugging Face Hub at startup and exposes a stateless
prediction endpoint.

## Endpoints

- `GET /` — health and model readiness
- `POST /api/predict` — classify `{ "text": "...", "threshold": 0.6 }`

## Space variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO_ID` | `shareeep/singbert-hatexplain-and-syntheticsghate` | Public HF model repo |
| `ALLOWED_ORIGINS` | `*` | Comma-separated frontend origins |
| `RATE_LIMIT_PREDICT_MAX` | `30` | Predictions allowed per IP/window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |

For central rate limiting, set `UPSTASH_REDIS_REST_URL` and
`UPSTASH_REDIS_REST_TOKEN` as Space secrets.
