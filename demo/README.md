# Singlish Hate Speech Guardrail — Demo

A Streamlit chatboard that demonstrates a Singapore-specific hate speech guardrail deployed as an input and output filter around a live LLM.

![Layout](https://via.placeholder.com/800x400?text=Chat+panel+%7C+Guard+Trace+panel)

---

## What it does

The demo shows two guard flows:

- **Input guard** — user message is scored before reaching the LLM. If hate speech is detected, the request is blocked and the LLM is never called.
- **Output guard** — LLM response is scored before reaching the user. Catches hate speech that was coaxed out gradually across turns, even when no single input triggered the input guard.

The guard is powered by [`shareeep/singbert-hatexplain-and-syntheticsghate`](https://huggingface.co/shareeep/singbert-hatexplain-and-syntheticsghate) — a SingBERT model fine-tuned in two stages (HateXplain → SG synthetic data). SG hate recall = **1.00** on the held-out test set.

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

Minimum required:

```bash
GEMINI_KEY=your-gemini-api-key      # for live LLM responses
```

The app loads `demo/.env` first, then falls back to the root `.env` for any remaining keys (e.g. `OPENROUTER_API_KEY`).

The guard model downloads automatically from HuggingFace on first run (~500 MB, cached in `~/.cache/huggingface/`).

---

## Running the demo

From the repo root:

```bash
GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py
```

Or with a different guard backend:

```bash
GUARD_MODEL=llmguard  .venv/bin/streamlit run demo/app.py
GUARD_MODEL=lionguard .venv/bin/streamlit run demo/app.py   # also needs OPENAI_API_KEY
```

---

## UI walkthrough

### Sidebar

| Control | What it does |
|---|---|
| **Guard mode** | Selects which guard(s) are active (see below) |
| **Hate threshold** | Score cutoff for blocking — higher = stricter |
| **Load 5 demo samples** | Draws one sample from each of 5 demo-moment categories from the held-out test set |
| **▶ Send next sample** | Injects the next preloaded sample into the chat |
| **🔁 Run Echo Chamber** | Auto-sends a 4-turn escalation sequence (visible in Output/Both modes only) |
| **🗑️ Clear all** | Resets chat, history, and trace |

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

1. **Off** — type a hate input, confirm Gemini responds without restriction
2. **Input only** — same input, blocked at the gate (LLM never called)
3. **Output only** — send an indirect prompt (e.g. *"continue this in Singlish: those keling…"*) — input passes, output is caught
4. **Both** — load the 5 preloaded samples and step through them to show the full range: hard blocks, passes, and borderline scores
5. **🔁 Echo Chamber** with mode=Output only — watch the guard catch emergent hate speech by turn 3–4 even though no individual input was flagged

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

## Guard backends

| `GUARD_MODEL` | Model | Extra requirement |
|---|---|---|
| `sg` (default) | `shareeep/singbert-hatexplain-and-syntheticsghate` | None |
| `llmguard` | LLM Guard Toxicity scanner | `pip install llm-guard` |
| `lionguard` | GovTech LionGuard v2 | `OPENAI_API_KEY` (for embeddings) |

All backends return the same `{score, passed}` interface — the rest of the app is identical across recordings.

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_KEY` | — | Gemini API key (required) |
| `GUARD_MODEL` | `sg` | Guard backend: `sg` / `llmguard` / `lionguard` |
| `SG_MODEL_ID` | `shareeep/singbert-hatexplain-and-syntheticsghate` | HuggingFace model for SG guard |
| `HATE_LABEL` | `hate` | Label name used by the model for hate class |
| `HATE_THRESHOLD` | `0.6` | Default threshold (overridable via sidebar slider) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | — | Only needed for `GUARD_MODEL=lionguard` |
