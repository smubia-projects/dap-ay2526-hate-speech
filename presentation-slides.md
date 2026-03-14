# DAP Hate Speech Project — Presentation Slides

---

## Slide 1: Intro

**Hate Speech Detection with Transformer Embeddings + Traditional ML**

11th DAP Project — SMUBIA

Dataset: **HateXplain** (15,383 train / 1,924 test)

---

## Slide 2: The Problem

**What:** Online hate speech is growing faster than human moderators can handle. Platforms need automated classifiers — but off-the-shelf solutions are expensive (LLMs) or fragile (keyword matching).

**Why it matters to me:**
- Wanted to understand what actually happens inside BERT/RoBERTa — not just use them as black boxes
- Curious whether you even need deep learning end-to-end, or if traditional ML can compete when given the right features
- A chance to go deep on embeddings, fine-tuning, and classifier mechanics — skills that transfer to any NLP problem

---

## Slide 3: Core Learning Goals

1. **Understand the embedding space** — what do 768-dimensional vectors actually look like, and how does fine-tuning change their geometry?
2. **Compare paradigms** — end-to-end deep learning vs frozen embeddings + traditional ML (XGBoost, Logistic Regression, Random Forest, MLP)
3. **Quantify the bottleneck** — is performance limited by the encoder, the classifier, or the data?
4. **Explore LLM approaches** — zero-shot, fine-tuned LLMs (Qwen, GPT-4o-mini, Claude Haiku) as alternative classifiers

---

## Slide 4: Show Us the Data — Raw Samples

**Input:** Social media posts (Twitter, Gab) — raw text, often noisy, slang-heavy, no punctuation.

**Output:** Classification label — `hatespeech`, `offensive`, or `normal` (3-class) / `hate` vs `non-hate` (binary)

| Post | Label |
|------|-------|
| "nothing royal about this wedding he is a bastard and she is a nignog halfbreed thot" | **hatespeech** |
| "caucasians wearing masks to prevent coronavirus and still don't wash their hands" | **offensive** |
| "i think immigrants should replace people like you" | **normal** |

The dataset also includes **annotator rationales** — highlighted words that annotators considered evidence:

> "diversity is a code word for **white genocide** immigrants contribute and integrate 3rd world **retards** freeload and fornicate"

3 annotators per post, majority vote decides the label. Rationale words are used for rationale-weighted pooling of embeddings.

> [IMAGE PLACEHOLDER: screenshot of a raw JSONL record showing post_tokens, annotator labels, rationale arrays, and target groups]

---

## Slide 5: Show Us the Data — Visualisation

**Class distribution** (roughly balanced, offensive slightly over-represented):

| Class | Train | Test |
|-------|------:|-----:|
| Hatespeech | 30.9% | 30.9% |
| Offensive | 40.6% | 40.6% |
| Normal | 28.5% | 28.5% |

> [IMAGE PLACEHOLDER: t-SNE plot from embedding-experiments.ipynb showing vanilla vs fine-tuned embedding clusters — 3 panels: vanilla (blob), fine-tuned binary (2 clusters), fine-tuned 3-class (3 clusters). This is the key visual — shows how fine-tuning reshapes the geometry.]

> [IMAGE PLACEHOLDER: bar chart of class distribution across train/val/test splits]

---

## Slide 6: Progress / Timeline

| Phase | Status |
|-------|--------|
| EDA + text length analysis | Done |
| BERT binary fine-tuned end-to-end (87.7% acc) | Done |
| RoBERTa 3-class fine-tuned end-to-end (70.0% acc) | Done |
| Embedding geometry analysis (silhouette, kNN, centroid) | Done |
| Vanilla embeddings + 4 classifiers | Done |
| Fine-tuned embeddings + 4 classifiers | Done |
| LLM experiments (zero-shot, API, fine-tuned adapter) | Done |
| Consolidated notebook + findings doc | Done |
| Cross-dataset eval (HateBenchSet) | **Next — Hack Day** |
| Localised hate speech (Singapore) | **Next — Hack Day** |

---

## Slide 7: Aha Moment #1 — Embeddings Are the Bottleneck

The biggest insight: **the quality of the embeddings determines the ceiling — the classifier barely matters.**

| Embeddings | Classifier | 3-Class Accuracy |
|---|---|---|
| Vanilla RoBERTa | Best classifier | 62.2% |
| Fine-Tuned RoBERTa | XGBoost | **70.3%** |
| Fine-Tuned RoBERTa | Logistic Regression | 68.8% |

Switching vanilla → fine-tuned: **+8% gain**.
Switching classifiers within fine-tuned: only **~2% spread**.

The embedding space went from silhouette ≈ 0.00 (classes indistinguishable) to ≈ 0.18 (clear clusters). That geometric shift is what makes everything work.

> [IMAGE PLACEHOLDER: side-by-side t-SNE — vanilla (flat blob) vs fine-tuned (separated clusters), with silhouette scores annotated]

---

## Slide 8: Aha Moment #2 — Traditional ML Can Match Deep Learning

| Approach | Binary Acc | 3-Class Acc |
|---|---|---|
| Frozen fine-tuned embeddings + XGBoost | **86.5%** | **70.3%** |
| End-to-end BERT/RoBERTa | **87.7%** | 70.0% |
| Zero-shot LLM (GPT-4o-mini) | 70.0% | — |
| Fine-tuned LLM adapter (Qwen 3B) | 76.5% | — |

For binary: within **1.2%** of end-to-end. For 3-class: embeddings + XGBoost **edges out** the end-to-end RoBERTa.

The encoder's single `nn.Linear(768, 3)` head is the weak link — XGBoost exploits the 768-dim space better.

And the LLMs? Zero-shot GPT-4o-mini scored 70% on binary — worse than a fine-tuned XGBoost on frozen embeddings. Fine-tuning a 3B parameter LLM with LoRA only got to 76.5%. The small encoder approach wins on both cost and accuracy.

---

## Slide 9: Aha Moment #3 — The "Normal" Class Problem

Every model — encoder, embedding+classifier, LLM — struggled with the same thing: **misclassifying normal text**.

RoBERTa end-to-end: normal recall only **45%** (over half of innocent posts flagged as harmful).

Example failures from the fine-tuned RoBERTa:

| Post | Predicted | Actual |
|------|-----------|--------|
| "This is a great movie, I really enjoyed it." | Offensive (92.8%) | Normal |
| "Just tried the new bakery on 5th street. The croissants are amazing!" | Offensive (92.0%) | Normal |
| "Shut up and sit down. Your opinions are worthless." | Normal (63.8%) | Offensive |

The model has an **offensive-class bias** — it defaults to "offensive" for anything ambiguous. This is the hardest boundary in the problem: where does strong opinion end and hate begin?

---

## Slide 10: Next Steps — Hack Day Plan

### Phase 1: Cross-Dataset Generalisation
- Run the existing pipeline on **HateBenchSet** (unseen English hate speech)
- Baseline: do models trained on HateXplain generalise at all?

### Phase 2: Localised Hate Speech (Singapore)
- Test on **Singaporean slang + code-switching** — distribution shift the models have never seen
- "this kind of people damn jialat sia" — does the model understand this?
- Two paths: **(a)** fine-tune LLM adapters on SG data, **(b)** continued MLM pre-training to teach encoders Singlish

### Phase 3: Evaluation
- Compare: vanilla → fine-tuned (HateXplain) → fine-tuned (HateXplain + SG)
- Does localised fine-tuning help without degrading original performance?

> [PLACEHOLDER: detailed hack day plan — data sources, tokenizer coverage check, binary vs 3-class decision]

---

## Slide 11: Next Steps — Blockers & Help Needed

**Key Gaps:**
- **SG data source** — need a Singlish corpus for continued MLM (HardwareZone forums? Reddit r/singapore? NUS SMS Corpus?)
- **Test set integrity** — SG hate speech test set should come from a different source than training data, otherwise we're just testing GenAI's writing style
- **Tokenizer coverage** — BERT's WordPiece may shred Singlish tokens into meaningless fragments → need to check before committing to the embedding path
- **Normal class recall** — still the weakest link across all approaches. Class weighting in XGBoost/MLP is pending.

**Bid for help:**
- Anyone with experience in **multilingual / code-switching NLP**?
- Access to or knowledge of **Singapore-specific text datasets** (licensed for research)
- Ideas on tackling the **offensive vs normal boundary** — the hardest classification edge

> [PLACEHOLDER: remaining DAP timeline with hack day milestones]
