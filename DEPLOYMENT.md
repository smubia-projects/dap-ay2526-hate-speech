# Deployment handoff

## Architecture

| Layer | Target | Directory |
|---|---|---|
| Model | Hugging Face Hub | `smubiainfrastructure/dap-ay2526-hate-speech-model` |
| API | Hugging Face Spaces, Docker SDK | `backend/` |
| Web app | Vercel, Vite preset | `frontend/` |

The canonical identifier for this project is its full folder name,
`dap-ay2526-hate-speech`. It is used unabbreviated as the HF Space name, the
Vercel project name, and the rate-limit engine slug. `Beat the Guard!` is the
public-facing product name only — it never appears as an infrastructure
identifier.

The guard model is a club-owned mirror of the fine-tune published at
`shareeep/singbert-hatexplain-and-syntheticsghate`. The mirror is what the Space
loads, so the demo does not depend on a personal account remaining public.

The API is stateless. Every request contains the complete message and policy
threshold; the only server-side state is the read-only model loaded at startup.
The session tally, strictness setting, and CTA counter all live in the browser.

## Local development

Backend:

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements_space.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

Frontend:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

## Hugging Face Space

**Prerequisite:** Hugging Face requires a PRO subscription on the owning account to
host a Docker Space on free `cpu-basic`. Creating one without it fails with
`402 Payment Required`. Spaces created before this policy took effect keep running,
so the account's older Spaces are not evidence that a new one can be created.

Create a public Space at `smubiainfrastructure/dap-ay2526-hate-speech` with the **Docker** SDK.
Push the contents of `backend/` to the Space repository root.

Set these Space variables:

| Variable | Value |
|---|---|
| `MODEL_REPO_ID` | `smubiainfrastructure/dap-ay2526-hate-speech-model` |
| `ALLOWED_ORIGINS` | `*` initially; replace after Vercel deployment |
| `RATE_LIMIT_PREDICT_MAX` | `30` |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` |

Optional rate-limit secrets:

- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

The model repository is public, so the Space does not need an HF token. No LLM
or embedding API keys are used by the deployed application.

Verify the Space:

```bash
curl https://smubiainfrastructure-dap-ay2526-hate-speech.hf.space/
curl -X POST https://smubiainfrastructure-dap-ay2526-hate-speech.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Aiyah, this queue damn rabak today.","threshold":0.6}'
```

The health response must contain `"status":"ok"` and
`"model_loaded":true`.

## Vercel

1. Import the GitHub repository in Vercel and name the project
   `dap-ay2526-hate-speech`.
2. Set **Root Directory** to `frontend`.
3. Set `VITE_API_BASE=https://smubiainfrastructure-dap-ay2526-hate-speech.hf.space`.
4. Deploy using the detected Vite build settings.
5. Replace `ALLOWED_ORIGINS=*` in the Space with the exact Vercel production
   URL and restart the Space.

## Central rate-limit registration

Project slug: `dap-ay2526-hate-speech`; action bucket: `predict`.

Add default/demo `predict` rows, a window row, and an enabled Status row to the
shared control sheet, then run **Sync limits to Redis** and
**Sync project on/off**. Without Upstash configuration, the shared engine fails
open and the demo remains available.

## Verification checklist

- Health reports `model_loaded: true` and `guards: { sg: true, regular: true }`.
- Prediction accepts one message and returns a `guards` array (SG-Tuned first,
  then Standard), each with probability, verdict, latency, and model repo ID.
- The two guards visibly diverge on the seeded probes: local-slur hate → SG
  blocks / Standard allows; harmless Singlish → SG allows / Standard blocks.
- First-load and prediction waiting states are visible while the free Space
  wakes up (two models load, so cold start is longer).
- A message can be scored by both guards without server-side session state.
- The strictness slider shifts verdicts across allow / review / block.
- The DAP CTA modal fires after 3 checks, is dismissible, and never blocks.
- HTTP 429 displays a slow-down notice; HTTP 503 displays a paused/warming
  notice.
- Frontend production build succeeds and CORS accepts only the Vercel origin.
