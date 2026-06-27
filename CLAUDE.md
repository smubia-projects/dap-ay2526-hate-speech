# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Environment

This project uses `uv` for dependency management with a `.venv` at the repo root.

```bash
uv venv          # first time
uv sync          # install / sync deps
.venv/bin/python3 <script>
.venv/bin/streamlit run <app>
```

The demo loads `demo/.env` first, then falls back to the root `.env`. Both are gitignored.

| Env var | What it does | When needed |
|---|---|---|
| `OPENROUTER_API_KEY` | Auths the LLM chat in `demo/pages/chat.py` (used via the `openai` Python client against `openrouter.ai`) | always, for live chat |
| `OPENROUTER_MODEL` | Which LLM to call (default `google/gemini-3-flash-preview`) | optional |
| `GUARD_MODEL` | Which guard backend to dispatch — `sg` / `llmguard` / `lionguard` (and future `lionguard-2.1` / `sea-guard`) | optional, default `sg` |
| `SG_MODEL_ID` | HF model for the SG guard (default `shareeep/singbert-hatexplain-and-syntheticsghate`) | optional |
| `HATE_LABEL` | Label name for hate class (default `hate`); cross-check `model.config.id2label` if predictions look off | optional |
| `HATE_THRESHOLD` | Default block threshold (default `0.6`) | optional |
| `OPENAI_API_KEY` | LionGuard 2.0 embeddings (`text-embedding-3-large`) | only if `GUARD_MODEL=lionguard` |
| `GEMINI_KEY` | LionGuard 2.1 embeddings (`gemini-embedding-001`) — planned backend | only if `GUARD_MODEL=lionguard-2.1` |
| `SEA_LION_API_KEY` | AISG SEA-Guard via `api.sea-lion.ai` — planned backend | only if `GUARD_MODEL=sea-guard` |

See `demo/example.env` for a working template.

## Running things

```bash
GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py
```

Notebooks live in:
- `notebooks/narrative_pre_midterm/` — embedding deep-dive (3 notebooks)
- `notebooks/narrative_finals/` — SingBERT / LoRA / Guardrails-AI experiments (6 notebooks)

Several are Colab-designed and contain `!pip install` and `from google.colab import userdata` cells — skip those locally.

The training corpus (`data/singlish/synthetic/20_terms/consolidated.jsonl`) is **gitignored**. A fresh clone won't have it; the demo will fail at `sample_loader.load_rows()` until the file is restored. `data/README.md` is the public-facing description (schema, 20 terms, sample rows). For reproducibility, the consolidated set can be recovered from `git show 5be47d9:data/singlish/...` or from `archive/data/singlish/synthetic/wip-files/`.

## Architecture

### Project arcs

Two arcs, both visible in the repo:

1. **Pre-midterm — embedding deep-dive**: how encoder embeddings carry hate-speech signal, and that traditional ML classifiers on top of those embeddings can do real work. Notebooks at `notebooks/narrative_pre_midterm/`. Written findings at `docs/findings-embedding-space.md`.

2. **Finals — production guardrail**: lightweight SG-finetuned SingBERT, fine-tuned in two stages (HateXplain → SG synthetic), deployed as the guard layer in a Streamlit chatboard. Notebooks at `notebooks/narrative_finals/` (winning fine-tune at `singbert_sg_experiments.ipynb`, comparative eval at `singbert_vs_hatexplain_test.ipynb`, integration story at `singlish_guardrails_demo.ipynb`). Result: SG hate recall = 1.00, macro F1 = 0.91.

LLM LoRA experiments (Llama-Guard-3-8B, SEA-LION) sit alongside in `notebooks/narrative_finals/llm_*.ipynb` — they motivate why the small encoder won.

### Canonical dataset

`data/singlish/synthetic/20_terms/consolidated.jsonl` — 533 rows, gitignored, the demo's sample source. Each row is `{text, label, target_group, source_term, source_category}`. Labels: `hate` (181) / `offensive` (165) / `normal` (187). Target groups: chinese / malay / indian / none.

The 60/20/20 split (319 train / 106 val / 108 test) is reproducible from the consolidated file using `archive/data/singlish/synthetic/wip-files/split_dataset_60_20_20.py`. Test-set metrics in `results/model_scores.csv` are the held-out 108 rows.

### Demo module layout (`demo/`)

| File | Role |
|---|---|
| `app.py` | Streamlit entry — sets page config + registers pages |
| `pages/chat.py` | Single-model chatboard with input / output / advisory guard modes |
| `pages/samples.py` | Browse and filter the consolidated dataset |
| `guard.py` | `run_guard(text, threshold)` — normalised `{score: float, passed: bool}` interface; dispatches on `GUARD_MODEL` |
| `singlish_validator.py` | `SinglishHateSpeech` — wraps any HuggingFace seq-classification model; exposes `_get_hate_prob()` and `validate()` |
| `sample_loader.py` | `get_demo_samples()` — one row from each of 5 demo-moment buckets sampled from the consolidated set |

The whole point of `guard.py` is that **guard backends are interchangeable.** Every backend returns the same `{score, passed}` dict, so the chat / arena / challenge pages don't care which model is loaded. Adding a backend = adding a branch in `load_guard()` + a branch in `run_guard()`. Current and planned:

| Backend | Status | Loads via | Returns |
|---|---|---|---|
| `sg` | live | HuggingFace `transformers` directly | binary hate-prob |
| `llmguard` | live | `llm_guard.input_scanners.Toxicity` | single toxicity score |
| `lionguard` (= 2.0) | live | `govtech/lionguard-2` + OpenAI embeddings | 11-category dict |
| `lionguard-2.1` | planned | `govtech/lionguard-2.1` + Gemini embeddings | 11-category dict |
| `sea-guard` | planned | OpenAI-compat call to `api.sea-lion.ai` | binary `SAFE`/`UNSAFE` |

### LLM integration

The chat page uses the `openai` Python client pointed at OpenRouter (`base_url="https://openrouter.ai/api/v1"`). Default model is `google/gemini-3-flash-preview` — chosen because it's known to produce Singlish hate speech under prompting, giving the output guard something to catch. Messages are stored as plain `{"role", "content"}` dicts in `st.session_state["gemini_history"]` (the name is historical; the API is OpenAI-shaped now). The `BLOCK_NONE` safety settings that older Gemini-direct code needed are not relevant here — OpenRouter passes prompts straight through. Refusals come back with `finish_reason == "content_filter"` and are surfaced as a "Model declined to answer" trace entry.

### Key models

- **Guard (primary)**: `shareeep/singbert-hatexplain-and-syntheticsghate` — pulled from HF on first run, cached in `~/.cache/huggingface/`.
- **LLM**: whatever `OPENROUTER_MODEL` points at; default `google/gemini-3-flash-preview`.

### Results tracking

`results/model_scores.csv` is the leaderboard data. The planned `demo/pages/leaderboard.py` will surface it inside the demo; the old `scripts/predictions_app.py` Streamlit viewer is at `archive/scripts/`.

## Gamification (planned, not yet built)

Design lives in `tmp/gamification-research/DESIGN-DOC.md` (gitignored, local-only). Three new pages on top of the current `chat.py` + `samples.py`:

- `pages/arena.py` — same input → all guard backends in parallel → 4-column scoreboard like the GovTech `lionguard-demo` HuggingFace Space
- `pages/challenge.py` — 5-round "you-as-moderator" game vs. the guards vs. ground truth
- `pages/leaderboard.py` — replace the archived `scripts/predictions_app.py` with an in-demo leaderboard

Backend additions (`lionguard-2.1`, `sea-guard`) land at the same time. See the design doc for build order.
