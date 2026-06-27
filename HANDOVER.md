# Handover — Singlish Hate Speech Guardrail Demo

## Project Description

This project builds a Singapore-specific hate speech guardrail for Singlish and local racialized language. The work has two connected arcs:

1. **Pre-midterm: embedding deep-dive**
   - Studied what encoder embeddings capture.
   - Compared embedding spaces and how they separate hate / offensive / normal text.
   - Showed that traditional ML classifiers can still work well when placed on top of strong embeddings.

2. **Finals: production-style guardrail**
   - Moved from analysis into a lightweight deployable safety layer.
   - Fine-tuned SingBERT into a Singapore-specific hate speech classifier.
   - Wrapped it as an input/output guardrail around a live LLM chat demo.
   - Added a gamified public-facing interface so users can try to bypass or compare the guardrails themselves.

The main validated model is:

```text
shareeep/singbert-hatexplain-and-syntheticsghate
```

It is a two-stage fine-tune: HateXplain first, then the SG synthetic Singlish hate speech dataset.

## Rationale

Generic hate speech systems often miss local context. Singaporean hate speech can be subtle, code-switched, and dependent on local slurs or reclaimed/offensive terms. A generic English toxicity model may over-block ordinary Singlish profanity while under-detecting protected-group targeting.

The project rationale is therefore:

- **Local language needs local guardrails.** Singlish and Singapore-specific slurs need cultural and linguistic coverage.
- **A lightweight classifier is useful in production.** Instead of relying only on a large LLM judge, the SG SingBERT guard can run as a fast input/output filter.
- **Guardrails should be testable by the public.** The gamified demo makes the model behavior visible: users can try examples, compare models, and see scores/latency instead of only reading final metrics.
- **The work connects research to deployment.** The pre-midterm embedding work explains why embeddings/classifiers are viable; the final demo shows how that becomes a usable safety layer.

## What Has Been Built

### Demo App

The Streamlit app lives in `demo/` and now has five pages:

- **Chat**: LLM chatboard with input guard, output guard, advisory mode, and trace panel.
- **Arena**: side-by-side comparison of all configured guard backends.
- **Challenge**: 5-round “you vs. the guards” moderation game.
- **Leaderboard**: model metrics plus session results.
- **Samples**: dataset browser for the local consolidated Singlish synthetic data.

Startup helper:

```bash
scripts/run_demo.sh
```

Validation helper:

```bash
.venv/bin/python3 scripts/validate_demo.py
```

### Validated Backends

| Backend | Status | Notes |
|---|---|---|
| `sg` | validated | Local SingBERT guard; primary demo path |
| `lionguard` | validated | GovTech LionGuard 2.0 + OpenAI embeddings |
| `lionguard-2.1` | validated | GovTech LionGuard 2.1 + Gemini embeddings |
| `sea-guard` | validated | AI Singapore SEA-Guard via SEA-LION API |
| `llmguard` | optional / blocked | Dependency conflict with current `transformers` stack |

Latest known useful validation artifacts:

```text
tmp/playwright-validation/20260628-024948/  # SG full UI regression
tmp/playwright-validation/20260628-025308/  # LionGuard direct + Arena UI validation
```

`tmp/` is gitignored, so these artifacts are local only.

## Hosting / Running / Repo Notes

### Repo

Branch:

```text
reorg/core-demo-narrative
```

Key files:

- `demo/app.py`: Streamlit entrypoint and page registration.
- `demo/guard.py`: swappable backend dispatcher.
- `demo/pages/`: Chat, Arena, Challenge, Leaderboard, Samples.
- `scripts/run_demo.sh`: interactive startup script with logging.
- `scripts/validate_demo.py`: Playwright validation script.
- `WIP.md`: current readiness and remaining tasks.

### Environment

Copy:

```bash
cp demo/example.env demo/.env
```

Important env vars:

```text
OPENROUTER_API_KEY      # live chat model
OPENROUTER_MODEL        # currently can point to DeepSeek v4 flash or another OpenRouter model
OPENAI_API_KEY          # LionGuard 2.0
GEMINI_KEY              # LionGuard 2.1
GEMINI_API_KEY          # alternate accepted name for LionGuard 2.1
SEA_LION_API_KEY        # SEA-Guard
SEA_GUARD_MODEL         # optional SEA-Guard model override
GUARD_MODEL             # sg / lionguard / lionguard-2.1 / sea-guard
```

Run:

```bash
GUARD_MODEL=sg scripts/run_demo.sh
```

Validate:

```bash
GUARD_API_TIMEOUT_SECONDS=8 .venv/bin/python3 scripts/validate_demo.py --backend sg
GUARD_API_TIMEOUT_SECONDS=8 .venv/bin/python3 scripts/validate_demo.py --backend lionguard --skip-ui
GUARD_API_TIMEOUT_SECONDS=8 .venv/bin/python3 scripts/validate_demo.py --backend lionguard-2.1 --skip-ui
GUARD_API_TIMEOUT_SECONDS=8 .venv/bin/python3 scripts/validate_demo.py --backend sea-guard --skip-ui
```

## Tech Stack

- Python 3.13 via `uv`
- Streamlit for the demo UI
- Playwright for end-to-end validation
- HuggingFace Transformers / Torch for SG SingBERT and LionGuard classifier heads
- OpenRouter for live chat model access
- OpenAI embeddings for LionGuard 2.0
- Google Gemini embeddings for LionGuard 2.1
- SEA-LION API for SEA-Guard
- Pandas / scikit-learn / notebooks for analysis and reporting

## Writeup Framing

### Project Description

This is a Singapore-specific hate speech detection and guardrail project. It begins with embedding analysis and traditional ML baselines, then becomes a production-style safety layer around LLM interactions. The final demo lets users test the guardrails directly through chat, model comparison, and a moderation game.

### Rationale

The project is motivated by the gap between generic toxicity models and local Singaporean language. Singlish, local slurs, and context-dependent offensive terms need a model that understands Singaporean usage. A lightweight guardrail model is also practical: it can be placed before and after an LLM without requiring every moderation decision to be made by another large LLM.

### Lessons Learnt / Takeaways

- Embeddings are not just abstract vectors; they can expose useful structure in language data.
- Traditional classifiers remain useful when paired with good embeddings.
- Generic hate speech models are not enough for local language and cultural context.
- A production-style safety system is more than model accuracy: it needs latency, failure handling, clean UX, validation scripts, and clear deployment docs.
- A gamified interface makes safety evaluation easier to understand. Letting users try to bypass the model is more memorable than only showing a score table.
- Multi-backend comparison is useful, but every backend has operational tradeoffs: API keys, latency, hosted model availability, dependency conflicts, and output schema differences.

## Presentation Framing

For slides, the narrative can be:

1. **Pre-midterm: understanding embeddings**
   - What do embeddings capture?
   - Can we classify hate speech using embeddings plus traditional ML?
   - What do embedding spaces reveal about offensive vs hate vs normal language?

2. **Finals: turning research into a guardrail**
   - We need a lightweight Singapore-specific model.
   - The model becomes an input/output safety layer around an LLM.
   - The demo makes behavior inspectable through traces, scores, latency, and model comparisons.

3. **Gamified public demo**
   - Users can try to bypass the guard.
   - Arena lets them compare SG SingBERT against LionGuard / SEA-Guard.
   - Challenge turns moderation into a game: user vs model vs ground truth.

## Remaining Work

See `WIP.md` for the active checklist. Main remaining items:

- Decide whether to isolate `llmguard` in a separate environment or leave it optional.
- Add persistent Arena feedback/vote logging.
- Add live latency aggregation to Leaderboard.
- Prepare a short presenter script for the final demo flow.
- Update root `README.md` repo tree/status if this branch is merged.
