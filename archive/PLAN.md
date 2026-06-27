# Singlish Hate Speech Detection — Experiment Plan

## Status Key
- [ ] Not started
- [~] In progress
- [x] Done

---

## Phase 0: Baselines (pre fine-tuning)

- [x] SingBERT zero-shot test — `notebooks/SingBERT/singbert_hatespeech_test.ipynb`
  - SingBERT (untrained head) vs dehatebert (2-class) on hand-picked Singlish examples
  - Result: SingBERT 38% accuracy, dehatebert misses all Singlish slurs

- [ ] **SingBERT vs HateXplain RoBERTa baseline** — `notebooks/SingBERT/singbert_vs_hatexplain_test.ipynb` ← NEW
  - SingBERT (untrained head) vs `shareeep/roberta-base-fine-tuned-hatexplain` (3-class)
  - Both models evaluated on hand-picked Singlish + synthetic test set (108 rows)
  - Error analysis: which Singlish terms does HateXplain RoBERTa miss?
  - Tokenisation comparison between the two tokenizers
  - **Run this first** — establishes the baselines that fine-tuning experiments improve upon

---

## Phase 1: Encoder Fine-Tuning Experiments

### 1a. SingBERT Experiments — `notebooks/SingBERT/singbert_sg_experiments.ipynb` (NEW)

Based on the structure of `notebooks/roberta/roberta_sg_experiments.ipynb`, adapted for BERT architecture:

- [ ] **Exp A: SingBERT → SG only**
  - Fine-tune `zanelim/singbert` with a 3-class head on the SG synthetic train split (319 rows)
  - Evaluate on SG test + HateXplain test

- [ ] **Exp B: SingBERT → HateXplain → SG (two-stage)**
  - Stage 1: Fine-tune SingBERT on HateXplain training data (learn hate speech patterns)
  - Stage 2: Fine-tune further on SG synthetic data (learn Singlish-specific patterns)
  - Evaluate on both test sets + check for catastrophic forgetting of HateXplain

- [ ] **Exp C: SingBERT + Combined training (HateXplain + SG oversampled)**
  - Merge HateXplain + SG training data (SG oversampled to ~10%)
  - Single-stage training
  - Evaluate on both test sets

### 1b. HateXplain RoBERTa → SG — already covered in existing notebook

- [x] `notebooks/roberta/roberta_sg_experiments.ipynb` — Exp A (SG-only), B (HateXplain→SG), C (combined)
  - Uses `shareeep/roberta-base-fine-tuned-hatexplain` as base
  - This is effectively the "HateXplain RoBERTa → SG fine-tune" experiment

### Cross-comparison

- [ ] **Results comparison table** — Compare best SingBERT experiment vs best RoBERTa experiment
  - Does Singlish pre-training (SingBERT) or hate speech pre-training (HateXplain RoBERTa) lead to better SG performance after fine-tuning?

---

## Phase 2: LLM LoRA Fine-Tuning

- [x] **Zero-shot / few-shot baselines** — `notebooks/llm/llm_sg_baselines.ipynb`
  - Tests Llama-Guard-3-8B, Claude Haiku, Gemini Flash, DeepSeek via OpenRouter
  - 3-class (hate/offensive/normal), Singlish-aware prompts
  - Both zero-shot and few-shot (2 examples per class)
  - Saves to `results/model_scores_sg.csv`

- [x] **Chat prompt + QLoRA fine-tuning** — `notebooks/llm/llm_lora_finetune_colab.ipynb`
  - Colab paid GPU (T4/A100), Unsloth + SFTTrainer
  - Singaporean content moderator system prompt (3-class)
  - **Model A:** `meta-llama/Llama-Guard-3-8B` (Part 2)
  - **Model B:** `aisingapore/sea-lion-7b-instruct` (Part 3, Unsloth with PEFT fallback)
  - GPU cleanup between models, leaderboard comparison in Part 4

- [ ] **Run notebooks** — execute on Colab, collect results

---

## Phase 3: SGHateCheck Evaluation — `notebooks/evaluation/evaluate_sghatecheck.ipynb` (NEW)

---

## Phase 4: Guardrails AI Integration — `notebooks/guardrails/singlish_guardrails_demo.ipynb` (DONE)

- [x] Custom `SinglishHateSpeech` validator wrapping any HuggingFace model
- [x] Baseline gap demo — show where ToxicLanguage fails on Singlish
- [x] Guard composition — SinglishHateSpeech + ToxicLanguage
- [x] Input guard + output guard pipeline demos
- [x] Benchmark on SG test set with threshold sensitivity analysis
- [x] Section 9: slot for swapping in fine-tuned model (zero code change)
- [x] Hub submission guide + project positioning
- [ ] **Swap in fine-tuned model** (after Phase 1 completes) — set `SG_FINETUNED_MODEL` in Section 9

## Phase 3: SGHateCheck Evaluation — `notebooks/evaluation/evaluate_sghatecheck.ipynb` (NEW)

- [ ] Source the SGHateCheck benchmark dataset + understand its label taxonomy
- [ ] Run inference with best encoder models (SingBERT, RoBERTa)
- [ ] Run inference with best LLM models
- [ ] Generate comparative report on Singapore-specific linguistic features

---

## Dataset Reference

**Location:** `data/singlish/synthetic/20_terms/`

| Split | Rows | Hate | Offensive | Normal |
|-------|------|------|-----------|--------|
| consolidated | 533 | 181 | 165 | 187 |
| train | 319 | 108 | 99 | 112 |
| val | 106 | 36 | 33 | 37 |
| test | 108 | 37 | 33 | 38 |

- 20 unique Singlish terms, all represented in every split
- Target groups: chinese, malay, indian, none
- Schema: `text`, `label`, `target_group`, `source_term`, `source_category`

---

## Priority Order

1. **Run Phase 0 baseline notebook** (quick, establishes reference points)
2. **Phase 1a: SingBERT fine-tuning experiments** (main new work)
3. **Phase 1 cross-comparison** (analyze encoder results)
4. Phase 2: LLM experiments (heavier compute, Colab dependent)
5. Phase 3: SGHateCheck (evaluation, depends on trained models)
