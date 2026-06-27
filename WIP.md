# WIP — Demo Readiness

Current status: **SG-only demo is ready.** The validated path is the local SingBERT guard (`GUARD_MODEL=sg`) through Chat, Arena, Challenge, Leaderboard, and Samples.

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
```

## Ready

- SG SingBERT downloads, loads, and runs locally.
- Chat blocks obvious hate and allows benign Singlish.
- Arena shows SG verdicts and handles unavailable backends cleanly.
- Challenge plays all 5 rounds to Game Over.
- Leaderboard and Samples render.
- Startup script writes logs under `tmp/streamlit-demo/`.
- Playwright validator starts/stops Streamlit and saves artifacts under `tmp/playwright-validation/`.

## Not Done Yet

### Backend Validation

- `llmguard`
  - Add `llm-guard` to dependencies if we want this backend live.
  - Validate one hate and one benign input.
  - Decide whether generic toxicity adds enough value for the demo.

- `lionguard`
  - Requires `OPENAI_API_KEY`.
  - Validate HF model load for `govtech/lionguard-2`.
  - Validate OpenAI embedding call and normalized 11-category output.
  - Measure first-call and warm-call latency.

- `lionguard-2.1`
  - Requires `GEMINI_KEY`.
  - Validate HF model load for `govtech/lionguard-2.1`.
  - Validate Gemini embedding shape matches `guard.predict(...)`.
  - Measure first-call and warm-call latency on available quota.

- `sea-guard`
  - Requires `SEA_LION_API_KEY`.
  - Confirm whether key access is available.
  - Validate SAFE / UNSAFE parsing.
  - If key access is not available, decide whether to replace this with an OpenRouter-hosted SEA-LION judge fallback.

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
