---
title: Beat the Guard API
emoji: "🛡️"
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---

# Beat the Guard API

CPU-only FastAPI inference for the **Beat the Guard** demo. The Space downloads two
public models from Hugging Face Hub at startup and scores every message with both, so
the frontend can put a Singapore-tuned guard and a generic one side by side.

- **SG-Tuned Guard** — `smubiainfrastructure/dap-ay2526-hate-speech-model` (a `hate` probability)
- **Standard Guard** — `s-nlp/roberta_toxicity_classifier`, a generic English toxicity foil (a `toxicity` probability)

## Endpoints

- `GET /` — health, model readiness, and per-guard load status
- `POST /api/predict` — classify `{ "text": "...", "threshold": 0.6 }`, returns a
  `guards` array (SG-Tuned first, then Standard) with each guard's probability, verdict
  (`allow` / `review` / `block`), latency, and repo id.

## Space variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO_ID` | `smubiainfrastructure/dap-ay2526-hate-speech-model` | SG-Tuned guard repo |
| `REGULAR_MODEL_REPO_ID` | `s-nlp/roberta_toxicity_classifier` | Standard toxicity foil repo |
| `ALLOWED_ORIGINS` | `*` | Comma-separated frontend origins |
| `RATE_LIMIT_PREDICT_MAX` | `30` | Predictions allowed per IP/window (anti-spam) |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |

The SG-Tuned guard is the product; if the Standard foil fails to load, `/api/predict`
simply omits it and the demo runs degraded rather than erroring.

For central rate limiting, set `UPSTASH_REDIS_REST_URL` and
`UPSTASH_REDIS_REST_TOKEN` as Space secrets.
