# dap-ay2526-hate-speech

Singapore-specific hate speech detection, built for the 11th DAP Project under SMUBIA.

A generic English toxicity filter fails at Singlish in two directions at once: it misses
hate carried by local slurs, and it over-blocks ordinary Singlish venting. This project
fine-tuned SingBERT into a Singapore-specific classifier that gets both right, then
deployed it as a guardrail you can try yourself.

Model: [`smubiainfrastructure/dap-ay2526-hate-speech-model`](https://huggingface.co/smubiainfrastructure/dap-ay2526-hate-speech-model)
— SingBERT fine-tuned in two stages (HateXplain → SG synthetic). Held-out SG hate
recall **1.00**, macro F1 **0.91**.

---

## What counts as what

The corpus separates **hate** from **offensive** deliberately, because a guardrail that
conflates them is unusable: it either blocks ordinary Singlish or lets targeted attacks
through.

| Label | Definition | Guard |
|---|---|---|
| **hate** | Attacks, dehumanises, or calls for the exclusion of a group defined by race, religion, ethnicity, or gender identity | blocked |
| **offensive** | Vulgar, insulting, or abusive, but aimed at an individual or a situation — no protected group is targeted | allowed |
| **normal** | Contains a loaded term used referentially, quoted, or descriptively | allowed |

**The test is who the words are aimed at**, not how rude they are:

> **hate** — *"Malay Singaporeans are all siao lang and should be locked up because they
> are a danger to society."*
>
> **offensive** — *"That CB kia cut my lane without signaling then still want to stare at
> me, really no manners at all."*
>
> **normal** — *"Apu neh neh is a slur that we should never use, really insensitive one."*

All three sentences carry Singlish vulgarity. Only the first targets a group.

### Why this matters to the guard

The guard is **binary** — it scores hate-ness only, so `offensive` and `normal` both
pass. That is the whole design: a moderation layer that blocks every vulgar message is
worse than no layer at all, because the false positives fall on ordinary users having
ordinary conversations. Terms including `ah kua`, `rabak`, `siao lang`, and `Zhu Bajie`
appear in the corpus under all three labels, so the term never decides the label —
context does. Getting that right is the model's actual job; catching unambiguous slurs
is the easy half.

The full label policy, the 20 source terms, and the construction pipeline are in
[`data/README.md`](data/README.md).

---

## Two surfaces

| | `demo/` — research | `backend/` + `frontend/` — public |
|---|---|---|
| Audience | The team | Anyone |
| UI | Streamlit, 5 pages | React one-pager, **Beat the Guard!** |
| Guards | 5 swappable backends | 2 fixed classifiers |
| Needs | 4 API keys + the gated corpus | nothing |
| Runs | Locally only | HF Space + Vercel |

**Beat the Guard!** puts the SG-tuned guard head-to-head with a generic English toxicity
model on the same message, so a visitor sees the gap rather than reading about it. It
makes no LLM, embedding, or paid API calls — two small classifiers on free CPU.

Design and verified model behaviour: [`docs/BEAT-THE-GUARD.md`](docs/BEAT-THE-GUARD.md).
Deploy runbook: [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Project arcs

**Pre-midterm — embedding deep-dive.** What encoder embeddings carry, and how far
traditional ML classifiers on top of them can go. Notebooks in
`notebooks/narrative_pre_midterm/`; written findings in
[`docs/findings-embedding-space.md`](docs/findings-embedding-space.md).

**Finals — production guardrail.** The embedding work established that the signal is
there; the finals turned it into a deployable safety layer. SingBERT fine-tuned on
HateXplain then on the SG synthetic corpus, wrapped as an input/output filter around a
live LLM chat, then opened up as a public comparison demo. Notebooks in
`notebooks/narrative_finals/`.

LoRA experiments on Llama-Guard-3-8B and SEA-LION sit alongside in
`notebooks/narrative_finals/llm_*.ipynb`. They are the reason the small encoder won: a
124M-parameter classifier that answers in ~50ms beats an 8B judge that needs a GPU, when
the task is one binary decision per message.

---

## Repo tree

```
dap-ay2526-hate-speech/
├── README.md · CLAUDE.md · DEPLOYMENT.md
├── pyproject.toml · uv.lock · .python-version · .gitignore
│
├── backend/                           ← deployed API (HF Space, Docker SDK)
│   ├── app.py                         · FastAPI, POST /api/predict (two guards)
│   ├── model_service.py               · CPU-only dual-classifier loader + inference
│   ├── rate_limit.py                  · adapter for the shared BIA rate-limit engine
│   ├── rate_limit_engine.py           · the shared engine itself
│   └── Dockerfile · requirements_space.txt · .dockerignore · README.md
│
├── frontend/                          ← deployed web app (Vercel, Vite + React)
│   ├── src/App.jsx                    · Beat the Guard!: one-page demo + about
│   ├── src/data/probes.js             · the seeded "try these" messages
│   ├── src/components/CtaModal.jsx    · DAP engagement CTA modal
│   └── src/utils/api.js               · backend client
│
├── demo/                              ← local research demo (Streamlit, multi-backend)
│   ├── app.py                         · Streamlit entry, page registration
│   ├── guard.py                       · swappable guard backend dispatcher
│   ├── singlish_validator.py          · SinglishHateSpeech wrapper
│   ├── sample_loader.py               · 5 demo-moment buckets from consolidated set
│   ├── example.env · README.md
│   └── pages/
│       ├── chat.py                    · chatboard with input/output/advisory guard modes
│       ├── arena.py                   · side-by-side backend comparison
│       ├── challenge.py               · 5-round moderation game
│       ├── leaderboard.py             · metrics + session results
│       └── samples.py                 · dataset browser
│
├── data/README.md                     · schema, the 20 source terms, sample rows
│                                        (the raw .jsonl is gitignored — see below)
├── notebooks/
│   ├── narrative_pre_midterm/         · embedding deep-dive (3 notebooks)
│   └── narrative_finals/              · SingBERT, LoRA, Guardrails-AI (6 notebooks)
├── results/model_scores.csv           · model comparison metrics
├── scripts/                           · demo startup + Playwright validation
├── docs/                              · findings, dataset decisions, reports
└── archive/                           ← superseded but recoverable
```

---

## Quick start

### The public demo, locally

```bash
# Backend
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements_space.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860

# Frontend, in another shell
cd frontend
cp .env.example .env
npm install && npm run dev
```

Both models download from HF Hub on first run (~1 GB total) and cache in
`~/.cache/huggingface/`. No API keys.

### The Streamlit research demo

```bash
uv venv && uv sync
cp demo/example.env demo/.env      # minimum: OPENROUTER_API_KEY
# restore data/singlish/synthetic/20_terms/consolidated.jsonl (gitignored)
GUARD_MODEL=sg scripts/run_demo.sh
```

Validate the UI end to end with `.venv/bin/python3 scripts/validate_demo.py`.

---

## Guard backends — swap, don't rewrite

`demo/guard.py` is a pure dispatch layer. Every backend returns the same normalised
result — `score`, `passed`, `latency_ms`, `backend`, plus optional `fine_grained`,
`verdict_label`, and `error` — so the demo pages don't change when you swap models.
Adding a backend is two branches: one in `load_guard()`, one in `run_guard()`.

| `GUARD_MODEL` | Source | Key needed |
|---|---|---|
| `sg` (default) | this project's SingBERT | none |
| `lionguard` (= 2.0) | `govtech/lionguard-2` + `text-embedding-3-large` | `OPENAI_API_KEY` |
| `lionguard-2.1` | `govtech/lionguard-2.1` + `gemini-embedding-001` | `GEMINI_KEY` |
| `sea-guard` | AISG SEA-Guard via `api.sea-lion.ai` | `SEA_LION_API_KEY` |
| `llmguard` | `llm-guard` Toxicity scanner | none |

Both LionGuard versions are drop-in equivalents — same 11-category output schema, same
normalised result. Pick whichever key you already have. `llmguard` is scaffolded but not
installable alongside the current stack: it pins `transformers==4.51.3` while this repo
needs `>=5.1.0`. It stays in the dispatcher for whoever wants to isolate it in a separate
environment.

The deployed **Beat the Guard!** backend deliberately uses neither — LionGuard and
SEA-Guard need API keys and carry 4–9s cold-start latency, which is acceptable in a
research tool and not in a public demo.

---

## Why the data is gated

The synthetic Singlish corpus contains racial slurs targeting specific Singaporean
communities. Every row is LLM-generated, but broad distribution still risks normalising
the language, so the corpus is **not in the public repo** — `.gitignore` excludes
`data/singlish/**/*.jsonl`.

What is shared: the model on HF Hub, [`data/README.md`](data/README.md) (schema, the
20-term table, sample rows, and the generation pipeline), and the held-out metrics in
`results/model_scores.csv`.

---

## What the project found

- Embeddings are not just abstract vectors — they expose structure that ordinary
  classifiers can use, which is what made the small-model path credible.
- Generic hate-speech models are not enough for local language. The failure is
  bidirectional, and the over-blocking half is the one people underestimate.
- A production safety layer is more than accuracy: it needs latency, failure handling,
  a clear UX, and an honest account of what it does not catch.
- Letting people try to bypass the guard teaches more than a metrics table does.
- Every guard backend carries operational trade-offs — keys, latency, hosted
  availability, output schema. The swappable interface exists so those trade-offs stay
  a deployment decision rather than a rewrite.

---

## Naming

The full folder name `dap-ay2526-hate-speech` is the canonical identifier, used
unabbreviated for the HF Space, the Vercel project, and the rate-limit engine slug.
**Beat the Guard!** is the public product name for the web demo only; it is never an
infrastructure identifier.
