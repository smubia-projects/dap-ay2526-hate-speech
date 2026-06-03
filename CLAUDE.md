# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses `uv` for dependency management with a `.venv` at the repo root.

```bash
# Create venv (first time)
uv venv

# Install / sync dependencies
uv sync

# Always invoke python/streamlit through the venv
.venv/bin/python3 <script>
.venv/bin/streamlit run <app>
```

The demo loads `demo/.env` first, then falls back to the root `.env`. Both are gitignored. Keys in use:
- `GEMINI_KEY` — Gemini API (google-genai SDK)
- `OPENROUTER_API_KEY` — OpenRouter LLM calls
- `GUARD_MODEL` — selects guard backend: `sg` | `llmguard` | `lionguard` (default: `sg`)
- `SG_MODEL_ID` — HuggingFace model ID for the SG guard (default: `shareeep/singbert-hatexplain-and-syntheticsghate`)
- `GEMINI_MODEL` — Gemini model name (default: `gemini-2.5-flash`)

See `demo/example.env` for a full template.

## Running things

```bash
# Guardrail chatboard demo (main deliverable)
GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py

# Model leaderboard dashboard
.venv/bin/streamlit run scripts/predictions_app.py

# Dataset viewer (with judge flags and manual edit)
.venv/bin/streamlit run scripts/view_dataset.py
```

Notebooks in `notebooks/finals-phase2/` are the canonical experiment notebooks — designed for Google Colab (they contain `!pip install` and `from google.colab import userdata` cells that must be skipped locally).

## Architecture

### Project phases

The work spans three phases documented in `PLAN.md`:

1. **Phase 1 — Encoder fine-tuning**: SingBERT and HateXplain RoBERTa fine-tuned on SG synthetic data. Best result: `shareeep/singbert-hatexplain-and-syntheticsghate` (two-stage: HateXplain → SG synthetic). SG hate recall = 1.00, macro F1 = 0.91.

2. **Phase 2 — LLM LoRA fine-tuning**: Llama-Guard-3-8B and SEA-LION fine-tuned via QLoRA on Colab (`notebooks/finals-phase2/llm/`). Results in `results/model_scores.csv`.

3. **Phase 3 (demo) — Guardrail chatboard**: The best Phase 1 model deployed as a Streamlit chatboard with input + output guards. Code lives in `demo/`.

### Canonical dataset

`data/singlish/synthetic/20_terms/split_60_20_20/` — the final held-out split (60/20/20). Each row is `{text, label, source_term, target_group}`. Labels: `hate` / `offensive` / `normal`. 108 test rows, never used in training.

### Demo module layout (`demo/`)

| File | Role |
|---|---|
| `app.py` | Streamlit entry point — layout, session state, guard/LLM orchestration |
| `guard.py` | `run_guard(text, threshold)` — normalised `{score, passed}` interface wrapping any backend |
| `singlish_validator.py` | `SinglishHateSpeech` class — wraps HuggingFace model, exposes `_get_hate_prob()` and `validate()` |
| `sample_loader.py` | `get_demo_samples()` — draws one row from each of 5 demo-moment buckets from the test split |

`guard.py` dispatches based on `GUARD_MODEL` env var. All three backends (`sg`, `llmguard`, `lionguard`) return the same `{score: float, passed: bool}` dict so `app.py` is backend-agnostic.

### Gemini integration

Uses `google-genai` SDK (`from google import genai`), **not** the older `google-generativeai`. The client is initialised once at module level in `app.py`. Gemini chat history is stored as `{"role": ..., "parts": [...]}` dicts in `st.session_state["gemini_history"]` and converted to `types.Content` objects on each call. Safety settings are set to `BLOCK_NONE` across all harm categories so the external guardrail is the only safety layer — this is intentional for the demo.

### Key models

- **Guard model (primary)**: `shareeep/singbert-hatexplain-and-syntheticsghate` — downloaded from HuggingFace on first run, cached in `~/.cache/huggingface/`.
- **LLM**: `gemini-2.5-flash` (default). `gemini-3-flash-preview` is also available and is more likely to generate Singlish hate speech under `BLOCK_NONE` (better for the output guard demo).

### Results tracking

`results/model_scores.csv` — all model evaluations. `scripts/predictions_app.py` visualises it as a leaderboard. New experiment results should be appended here.
