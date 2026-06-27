# Singlish Hate Speech Guardrail — Demo

A Streamlit chatboard that demonstrates a Singapore-specific hate speech guardrail deployed as an input and output filter around a live LLM.

![Layout](https://via.placeholder.com/800x400?text=Chat+panel+%7C+Guard+Trace+panel)

---

## What it does

The demo shows two guard flows:

- **Input guard** — user message is scored before reaching the LLM. If hate speech is detected, the request is blocked and the LLM is never called.
- **Output guard** — LLM response is scored before reaching the user. Catches hate speech that was coaxed out gradually across turns, even when no single input triggered the input guard.

The validated demo path is powered by [`shareeep/singbert-hatexplain-and-syntheticsghate`](https://huggingface.co/shareeep/singbert-hatexplain-and-syntheticsghate) — a SingBERT model fine-tuned in two stages (HateXplain → SG synthetic data). SG hate recall = **1.00** on the held-out test set.

LionGuard 2.0, LionGuard 2.1, and SEA-Guard are validated as API-backed comparison backends when their keys are configured. `llmguard` remains optional because its dependency stack conflicts with the current `transformers` version. See [`../WIP.md`](../WIP.md).

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/shareeep/11th-DAP-hate-speech
cd 11th-DAP-hate-speech

# Create venv and install all dependencies
uv venv
uv sync
```

**2. Configure environment**

Copy `demo/example.env` to `demo/.env` and fill in your keys — this file is gitignored and only used by the demo:

```bash
cp demo/example.env demo/.env
# then edit demo/.env
```

Minimum required for live LLM responses:

```bash
OPENROUTER_API_KEY=your-openrouter-key
```

Without `OPENROUTER_API_KEY`, the chat page uses an `[LLM STUB]` response. The local SG guard still works and the Playwright validation intentionally accepts the stub path.

The app loads `demo/.env` first, then falls back to the root `.env` for any remaining keys.

The guard model downloads automatically from HuggingFace on first run (~500 MB, cached in `~/.cache/huggingface/`).

---

## Running the demo

From the repo root:

```bash
scripts/run_demo.sh
```

This prompts for a guard backend, starts Streamlit, and writes logs to
`tmp/streamlit-demo/<backend>-<timestamp>.log` plus `tmp/streamlit-demo/latest.log`.
It also disables Streamlit's file watcher by default to avoid noisy optional
`transformers` / `torchvision` import traces. Override defaults when needed:

```bash
GUARD_MODEL=sg PORT=8501 scripts/run_demo.sh
FILE_WATCHER_TYPE=auto scripts/run_demo.sh
```

Manual equivalent:

```bash
GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py --server.fileWatcherType none
```

Or with a different guard backend:

```bash
GUARD_MODEL=llmguard      scripts/run_demo.sh   # currently optional; dependency conflict noted in WIP.md
GUARD_MODEL=lionguard     scripts/run_demo.sh   # needs OPENAI_API_KEY
GUARD_MODEL=lionguard-2.1 scripts/run_demo.sh   # needs GEMINI_KEY
GUARD_MODEL=sea-guard     scripts/run_demo.sh   # needs SEA_LION_API_KEY
```

---

## UI walkthrough

### Sidebar

| Control | What it does |
|---|---|
| **Guard mode** | Selects which guard(s) are active (see below) |
| **Hate threshold** | Score cutoff for blocking — higher = stricter |
| **Load 5 demo samples** | Draws one sample from each of 5 demo-moment categories from the held-out test set |
| **Send next sample** | Injects the next preloaded sample into the chat |
| **Preset prompts / Send preset** | Sends one curated prompt, including the four-step echo-chamber sequence |
| **Clear all** | Resets chat, history, and trace |

### Guard modes

| Mode | Behaviour |
|---|---|
| **Both** | Input blocked before LLM call; output filtered after — production mode |
| **Input only** | Only the user's message is screened |
| **Output only** | User message passes through; LLM response is screened |
| **Advisory** | Nothing is blocked — flagged messages accumulate in the moderation queue in the sidebar |
| **Off** | No guardrail — LLM responds freely |

### Guard trace panel

Every turn shows:
- `INPUT` score + colour-coded bar (green < 0.4 / amber < threshold / red ≥ threshold)
- `OUTPUT` score + bar
- Action label (`INPUT BLOCKED` or `OUTPUT FILTERED`) if triggered

---

## Recommended demo sequence

1. **Off** — type a hate input, confirm the LLM or stub responds without guard blocking
2. **Input only** — same input, blocked at the gate (LLM never called)
3. **Output only** — send an indirect prompt (e.g. *"continue this in Singlish: those keling…"*) — input passes, output is caught
4. **Both** — load the 5 preloaded samples and step through them to show the full range: hard blocks, passes, and borderline scores
5. **Echo chamber presets** with mode=Output only — send presets `[1/4]` through `[4/4]` to test whether the output guard catches escalation

---

## Validation

Run the Playwright validation from the repo root:

```bash
.venv/bin/python3 scripts/validate_demo.py
```

By default it starts Streamlit on port 8765 with `GUARD_MODEL=sg`, disables the
file watcher, validates Chat / Arena / Challenge / Leaderboard / Samples, then
stops the server. Artifacts are written to `tmp/playwright-validation/`.

Useful variants:

```bash
.venv/bin/python3 scripts/validate_demo.py --headed
.venv/bin/python3 scripts/validate_demo.py --base-url http://localhost:8501
.venv/bin/python3 scripts/validate_demo.py --backend lionguard --port 8766
```

---

## Preloaded sample categories

| Category | What it demonstrates |
|---|---|
| Clear hate, racial target | Unambiguous — guard fires hard |
| Singlish-specific slurs | Terms like *apu neh neh*, *cina babi* — Singlish slurs the guard must catch |
| Offensive, not hate | Singlish expletives — guard should pass, showing offensive ≠ hate |
| False positive trap | Normal sentence containing a known-bad term — tests context sensitivity |
| Borderline / grey zone | Scores in 0.4–0.65 range — motivates threshold discussion |

---

## Guard backends — swappable

`guard.py` is a thin dispatch layer; every backend returns a normalized result with `score`, `passed`, `latency_ms`, `backend`, and optional `fine_grained`, `verdict_label`, and `error` fields.

| `GUARD_MODEL` | Model | Extra requirement | Status |
|---|---|---|---|
| `sg` (default) | `shareeep/singbert-hatexplain-and-syntheticsghate` | none | validated |
| `llmguard` | LLM Guard Toxicity scanner | add `llm-guard` dependency | scaffolded, currently unavailable |
| `lionguard` (= 2.0) | [`govtech/lionguard-2`](https://huggingface.co/govtech/lionguard-2) + `text-embedding-3-large` | `OPENAI_API_KEY` | validated |
| `lionguard-2.1` | [`govtech/lionguard-2.1`](https://huggingface.co/govtech/lionguard-2.1) + `gemini-embedding-001` | `GEMINI_KEY` / `GEMINI_API_KEY` | validated |
| `sea-guard` | AISG SEA-Guard via `api.sea-lion.ai` | `SEA_LION_API_KEY` | validated |

**LionGuard 2 vs 2.1 — pick the one whose key you already have.** They produce the same 11-category output schema; the only difference is the embedding backbone (OpenAI vs Gemini). Either is a fine reference comparison for the SG-finetuned model.

Adding a new backend is a two-branch change in `guard.py` — one in `load_guard()` to import + initialize, one in `run_guard()` to normalise the output.

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter key for live chat responses; app uses a stub when missing |
| `GUARD_MODEL` | `sg` | Guard backend: `sg` / `llmguard` / `lionguard` / `lionguard-2.1` / `sea-guard` |
| `SG_MODEL_ID` | `shareeep/singbert-hatexplain-and-syntheticsghate` | HuggingFace model for SG guard |
| `HATE_LABEL` | `hate` | Label name used by the model for hate class |
| `HATE_THRESHOLD` | `0.6` | Default threshold (overridable via sidebar slider) |
| `OPENROUTER_MODEL` | `google/gemini-3-flash-preview` | OpenRouter model name |
| `OPENAI_API_KEY` | — | Only needed for `GUARD_MODEL=lionguard` |
| `GEMINI_KEY` / `GEMINI_API_KEY` | — | Only needed for `GUARD_MODEL=lionguard-2.1` |
| `SEA_LION_API_KEY` | — | Only needed for `GUARD_MODEL=sea-guard` |
| `SEA_GUARD_MODEL` | `aisingapore/SEA-Guard` | SEA-Guard API model name override |
