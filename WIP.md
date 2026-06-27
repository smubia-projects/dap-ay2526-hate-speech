# WIP — Demo Readiness

Current status: **SG + API-backed comparison demo is ready.** The local SingBERT guard (`GUARD_MODEL=sg`) is validated through Chat, Arena, Challenge, Leaderboard, and Samples. LionGuard 2.0, LionGuard 2.1, and SEA-Guard also pass direct backend validation with the configured keys.

Use:

```bash
GUARD_MODEL=sg scripts/run_demo.sh
```

Validate:

```bash
.venv/bin/python3 scripts/validate_demo.py
```

Last known passing validation:

```text
tmp/playwright-validation/20260628-010735/
tmp/playwright-validation/20260628-024948/  # SG full UI regression with live OpenRouter config
tmp/playwright-validation/20260628-025308/  # LionGuard direct + Arena UI validation
```

## Ready

- SG SingBERT downloads, loads, and runs locally.
- Chat blocks obvious hate and allows benign Singlish.
- Arena shows SG verdicts and handles unavailable backends cleanly.
- Challenge plays all 5 rounds to Game Over.
- Leaderboard and Samples render.
- Startup script writes logs under `tmp/streamlit-demo/`.
- Playwright validator starts/stops Streamlit and saves artifacts under `tmp/playwright-validation/`.
- LionGuard 2.0 validates with `OPENAI_API_KEY`.
- LionGuard 2.1 validates with `GEMINI_KEY` / `GEMINI_API_KEY`.
- SEA-Guard validates with `SEA_LION_API_KEY`.

## Backend Status

### Optional / Blocked

- `llmguard`
  - Add `llm-guard` to dependencies if we want this backend live.
  - Current environment note: `uv add llm-guard==0.3.16` is unsatisfiable because `llm-guard` pins `transformers==4.51.3` while this repo requires `transformers>=5.1.0`.
  - `uv add llm-guard` without a pin selects `llm-guard==0.3.15` and fails on Python 3.13 because `tiktoken==0.7.0` needs a Rust source build.
  - Treat as optional unless we either isolate it in a separate environment or intentionally downgrade `transformers` and revalidate SG/LionGuard.
  - Validate one hate and one benign input.
  - Decide whether generic toxicity adds enough value for the demo.

### Validated API Backends

- `lionguard`
  - Requires `OPENAI_API_KEY`.
  - Validated HF model load for `govtech/lionguard-2`.
  - Validated OpenAI embedding call and normalized 11-category output.
  - Observed direct-check latency: first hate call ~4.9-8.3s including model/API warm-up; benign warm call ~0.7-0.8s.

- `lionguard-2.1`
  - Requires `GEMINI_KEY` or `GEMINI_API_KEY`.
  - Validated HF model load for `govtech/lionguard-2.1`.
  - Validated Gemini embedding shape matches `guard.predict(...)`.
  - Observed direct-check latency: first hate call ~9.1s including model/API warm-up; benign warm call ~0.5s.

- `sea-guard`
  - Requires `SEA_LION_API_KEY`.
  - `SEA_GUARD_MODEL` can override the default model name.
  - Confirmed key access is available.
  - Validated SAFE / UNSAFE parsing.
  - Observed direct-check latency with `GUARD_API_TIMEOUT_SECONDS=8`: hate ~1.5s, benign ~6.7s.

### Demo Polish

- Add persistent feedback/vote logging for Arena, likely under `tmp/feedback.csv` or `tmp/arena_votes.csv`.
- Add live latency aggregation to Leaderboard after backend validation.
- Decide whether Challenge round-by-round breakdown should be expanded by default.
- Re-run Playwright validation after any UI changes.

### Documentation

- Keep `demo/README.md` status language aligned with actual validation.
- Update `demo/example.env` if backend requirements change.
- Add a short presenter script once the final backend set is known.

## Commit Guidance

Suggested split:

1. `feat(guard): add multi-backend guard dispatch`
2. `feat(demo): add arena challenge leaderboard pages`
3. `chore(demo): add startup and validation scripts`
4. `docs(demo): document validated and pending backend paths`
