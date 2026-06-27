# 11th-DAP-hate-speech

Singapore-specific hate speech detection for the 11th DAP Project under SMUBIA. Two arcs:

1. **Pre-midterm — embedding deep-dive.** What encoder embeddings carry, and how far traditional ML classifiers on top of them can go. Notebooks in `notebooks/narrative_pre_midterm/`.
2. **Finals — production guardrail.** A lightweight SG-finetuned SingBERT deployed as a swappable input + output filter in a Streamlit chatboard. Notebooks in `notebooks/narrative_finals/`; the live product is in `demo/`.

Best model: [`shareeep/singbert-hatexplain-and-syntheticsghate`](https://huggingface.co/shareeep/singbert-hatexplain-and-syntheticsghate) — SingBERT fine-tuned two-stage (HateXplain → SG synthetic). SG hate recall = **1.00**, macro F1 = **0.91** on the held-out test set.

---

## Repo tree

```
11th-DAP-hate-speech/
├── README.md
├── CLAUDE.md
├── pyproject.toml · uv.lock · .python-version · .gitignore
│
├── demo/                              ← the product
│   ├── app.py                         · Streamlit entry, page registration
│   ├── guard.py                       · swappable guard backend dispatcher
│   ├── singlish_validator.py          · SinglishHateSpeech wrapper
│   ├── sample_loader.py               · 5 demo-moment buckets from consolidated set
│   ├── example.env · README.md
│   └── pages/
│       ├── chat.py                    · LIVE: chatboard with input/output/advisory guard modes
│       ├── arena.py                   · LIVE: side-by-side backend comparison
│       ├── challenge.py               · LIVE: 5-round moderation game
│       ├── leaderboard.py             · LIVE: metrics + session results
│       └── samples.py                 · LIVE: dataset browser
│
├── data/
│   └── README.md                      · schema, the 20 source terms, sample rows
│                                        (the raw .jsonl is gitignored — see below)
│
├── notebooks/
│   ├── narrative_pre_midterm/         · embedding deep-dive (3 notebooks)
│   └── narrative_finals/              · SingBERT, LoRA, Guardrails-AI (6 notebooks)
│
├── results/
│   └── model_scores.csv               · leaderboard data
│
├── docs/
│   ├── findings-embedding-space.md
│   ├── findings-roberta-finetune.md
│   ├── synthetic_dataset_repair_decisions.md
│   ├── IS469_Shariff_Indiv_Report.pdf
│   └── IS469_Shariff_Indiv_Presentation_Slides.pdf
│
└── archive/                           ← superseded but recoverable
    ├── data/        ← hatexplain, hatebenchset, wip-files, raw scrapes
    ├── notebooks/   ← bert, llm, roberta, synthetic, early archive
    ├── scripts/     ← scrapers, generators, the old predictions_app.py
    ├── docs/        ← stale planning docs
    └── demo/        ← original plan / steps / raw idea
```

---

## Quick start

```bash
# 1. Create venv (Python 3.13) and install deps
uv venv
uv sync

# 2. Provide your keys
cp demo/example.env demo/.env
# edit demo/.env — minimum: OPENROUTER_API_KEY

# 3. Restore the dataset locally (it is gitignored)
#    Drop your consolidated.jsonl at
#    data/singlish/synthetic/20_terms/consolidated.jsonl

# 4. Launch the demo
GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py
```

The SG guard model (~500 MB) downloads from HuggingFace on first launch and caches in `~/.cache/huggingface/`.

---

## Why the data is gated

The synthetic Singlish corpus contains racial slurs targeting specific Singaporean communities. Even though every row is LLM-generated, the corpus is **not part of the public repo** — the `.gitignore` excludes `data/singlish/**/*.jsonl`. What's shared: the [model on HuggingFace](https://huggingface.co/shareeep/singbert-hatexplain-and-syntheticsghate), `data/README.md` (schema + 20-term table + sample rows), and the held-out test metrics in `results/model_scores.csv`.

---

## Guard backends — swap, don't rewrite

`demo/guard.py` is a pure dispatch layer. Every backend implements the same `{score: float, passed: bool}` contract, so the demo pages don't change when you swap models.

| `GUARD_MODEL` | Source | Status | Key needed |
|---|---|---|---|
| `sg` (default) | `shareeep/singbert-hatexplain-and-syntheticsghate` | validated | none |
| `llmguard` | `llm-guard` Toxicity scanner | optional / blocked | none, but dependency conflict |
| `lionguard` (= 2.0) | `govtech/lionguard-2` + `text-embedding-3-large` | validated | `OPENAI_API_KEY` |
| `lionguard-2.1` | `govtech/lionguard-2.1` + `gemini-embedding-001` | validated | `GEMINI_KEY` / `GEMINI_API_KEY` |
| `sea-guard` | AISG SEA-Guard via `api.sea-lion.ai` | validated | `SEA_LION_API_KEY` |

Both LionGuard versions are valid drop-ins — 2.0 if you have an OpenAI key, 2.1 if you have a Gemini key. Same output schema (11 categories), same `{score, passed}` after normalisation. Pick whichever key you already have.

---

## What's next

The gamified demo extension is now built on `reorg/core-demo-narrative`. See `HANDOVER.md` for the project narrative and current state, and `WIP.md` for remaining polish.
