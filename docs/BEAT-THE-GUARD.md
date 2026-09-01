# Beat the Guard! — build spec

The public-facing web demo for `dap-ay2526-hate-speech`. **Beat the Guard!** is the
product name; `dap-ay2526-hate-speech` remains the canonical infrastructure identifier
(HF Space, Vercel project, rate-limit slug).

## The idea

One input, two models scoring in parallel:

- **SG-Tuned Guard** — this project's fine-tuned SingBERT (`hate` probability).
- **Standard Guard** — a generic English toxicity model, as a baseline foil
  (`toxicity` probability).

The visitor types a message and sends it to both guards. Two bars fill. The point lands
in one second: a generic English filter fails at Singlish in **two** ways at once, and
the small SMU-trained model gets both right.

The two failure modes were verified directly against both models (see the probe battery
below). Hate must be expressed through **local slurs without an English swear word** for
the contrast to show — a generic model catches "cina *pigs*" via the English word, but is
blind to "apu neh neh should all go back to india".

| What happens (verdicts) | Headline | The lesson |
|---|---|---|
| SG blocks, Standard allows | "Ours caught it. The generic model missed." | Generic filters are blind to local-slur hate — the recruiting moment. |
| SG allows, Standard blocks | "The generic model over-blocked. Ours stayed calm." | Generic filters over-block ordinary Singlish slang/vulgarity as toxic. |
| Both block | "Both guards blocked it." | Carries an English swear word, so even the baseline catches it. |
| Both allow | "Both let it through." | Clean, or subtle enough that neither fires. |

Framing is **red team, not troll**: "Catch the model out." The SG guard resists both
failure modes — that's the flex; the generic model beside it fails constantly.

### Verified behaviour (threshold 0.6)

| Message | SG-Tuned (`hate`) | Standard (`toxicity`) |
|---|---|---|
| "these apu neh neh should all go back to india" | 0.99 **block** | 0.03 allow (miss) |
| "keling kia all cannot be trusted, always sneaky" | 0.83 **block** | 0.01 allow (miss) |
| "this driver damn cb, made me late again" | 0.01 allow | 1.00 **block** (over-block) |
| "aiyah this queue damn rabak today" | 0.00 allow | 0.99 **block** (over-block) |
| "these people are lazy and dirty, go back to your country" | 0.99 **block** | 1.00 **block** |
| "calling people keling is racist, we must stop it" | 0.00 allow | 0.00 allow |

The SG model reads structure, not keywords: "apu neh neh" alone scores `normal` 0.95,
but "apu neh neh should all go back" scores `hate` 0.99.

## Why this project exists (showcase copy)

Hate speech online isn't always loud. In Singapore it's often code-switched, misspelt on
purpose, or hidden in slang a generic model has never seen. A filter that only knows
textbook English over-blocks harmless Singlish and under-catches real local hate. **Local
language needs a local guardrail.** This DAP project fine-tuned SingBERT into a
Singapore-specific hate-speech classifier and puts it head-to-head with a standard model
so anyone can see the gap themselves.

Held-out SG metrics: hate recall **1.00**, macro F1 **0.91**.

## Architecture — DAP Pattern A (unchanged)

Model on HF Hub · stateless Docker backend on HF Spaces (port 7860, uid 1000) · React
frontend on Vercel. No LLM, embedding, or paid API calls. No database — all interaction
state lives in the browser. Both classifiers download from HF Hub at startup and run on
free CPU.

## API contract

```
POST /api/predict
  { "text": "...", "threshold": 0.6 }   # threshold in [0.2, 0.95]

200 OK
{
  "text": "...",
  "threshold": 0.6,
  "guards": [
    { "key": "sg",      "name": "SG-Tuned Guard", "metric": "hate",
      "probability": 0.94, "verdict": "block", "latency_ms": 120.0,
      "model_repo_id": "smubiainfrastructure/dap-ay2526-hate-speech-model" },
    { "key": "regular", "name": "Standard Guard", "metric": "toxicity",
      "probability": 0.08, "verdict": "allow", "latency_ms": 90.0,
      "model_repo_id": "s-nlp/roberta_toxicity_classifier" }
  ]
}
```

- `guards` is ordered: SG-Tuned first, then Standard. The SG-Tuned guard is the product;
  if the Standard foil fails to load, the array simply omits it and the demo runs
  degraded rather than erroring.
- `verdict` bands: `block` when `probability >= threshold`, `review` when
  `>= threshold - 0.2`, else `allow`.
- The outcome headline is computed **client-side** from the two verdicts — the backend
  stays dumb and honest.

```
GET /
  { "status": "ok", "model_loaded": true, "guards": { "sg": true, "regular": true } }
```

`model_loaded` mirrors the SG guard so the existing health-check and frontend
cold-start ping keep working.

## Backend env vars

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO_ID` | `smubiainfrastructure/dap-ay2526-hate-speech-model` | SG-Tuned guard repo |
| `REGULAR_MODEL_REPO_ID` | `s-nlp/roberta_toxicity_classifier` | Standard foil repo |
| `ALLOWED_ORIGINS` | `*` | CORS origins (set to the Vercel URL in prod) |
| `RATE_LIMIT_PREDICT_MAX` | `30` | Anti-spam: predictions per IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `UPSTASH_REDIS_REST_URL` / `_TOKEN` | — | Central rate limiting; fails open if unset |

## Frontend

Single-page React (Vite), themed as a **kopitiam face-off** — a warm light palette
(Manrope + Fraunces serif). Two views: the **demo** (default, no landing hero) and
**about** (how it works + stats + DAP recruit). The demo is a versus bout:

- the two guards flank a center "chit" holding the message — Generic filter (left),
  SG-tuned guard (right, home corner);
- each guard's verdict is a **traffic light** (green allow / amber review / red block)
  with a small probability + latency underneath (lamp is the primary signal, the number
  keeps the model credibility);
- one message box + "Send it in" below, with tappable **"try these" probe chips**
  (`data/probes.js`) that seed the box so nobody freezes — one per bucket from
  `data/README.md`'s taxonomy: local-slur hate, coded racism, Singlish vulgarity,
  harmless Singlish, obvious hate, counter-speech;
- an outcome banner in the ring, tinted by the table above;
- a **scorecard** of the generic model's two failure modes: **generic missed** (SG blocks,
  Standard allows) and **generic over-blocked** (SG allows, Standard blocks);
- a compact strictness (threshold) control — present for the model story, ignorable.

Engagement CTA modal (DAP playbook): fires client-side after 3 successful checks, keyed
`dap-ay2526-hate-speech:cta-shown`, never on a 429, always dismissible. Links: Join DAP,
GitHub, other projects.

Backend cold start is 10–20s on free CPU — the UI shows a waking state and pings `/` on
load.
