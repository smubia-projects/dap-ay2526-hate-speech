# Embedding Space Analysis: Findings & Discussion

## Executive Summary

We used BERT and RoBERTa as embedding models to generate rich 768-dimensional text representations, then fed those embeddings into traditional ML classifiers — Logistic Regression, XGBoost, Random Forest, and MLP — to classify hate speech. The question: can embeddings + traditional ML match or rival end-to-end deep learning?

**Dataset:** HateXplain — 15,383 train / 1,924 test samples.

### Binary Classification (Hate vs Non-Hate)

| Embeddings | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned BERT [rationale] | XGBoost | **86.5%** | **0.840** |
| Fine-Tuned BERT [rationale] | MLP (256→128→64) | 85.7% | 0.835 |
| Fine-Tuned BERT [rationale] | Random Forest | 85.6% | 0.827 |
| Fine-Tuned BERT [rationale] | Logistic Regression | 84.7% | 0.829 |
| Fine-Tuned BERT [cls] | XGBoost | 84.5% | 0.817 |
| Fine-Tuned BERT [cls] | Logistic Regression | 82.9% | 0.811 |
| _End-to-end BERT + classification head_ | _(reference)_ | _87.7%_ | _0.874_ |

The best embedding + ML setup (XGBoost on rationale-pooled embeddings) reaches **86.5% accuracy** — within **1.2%** of the end-to-end BERT fine-tuning baseline. All four classifiers land between 82.9% and 86.5%, a spread of just ~3.6%.

### 3-Class Classification (Hate / Offensive / Normal)

| Embeddings | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned RoBERTa [cls] | XGBoost | **70.3%** | **0.691** |
| Fine-Tuned RoBERTa [cls] | Random Forest | 70.0% | 0.686 |
| Fine-Tuned RoBERTa [cls] | Logistic Regression | 68.8% | 0.681 |
| Fine-Tuned BERT [cls] | XGBoost | 69.5% | 0.684 |
| Fine-Tuned RoBERTa [cls] | MLP (256→128→64) | 69.1% | 0.676 |
| Vanilla RoBERTa [cls] | Logistic Regression | 62.2% | 0.610 |
| Vanilla BERT [cls] | MLP (256→128→64) | 58.9% | 0.567 |
| _End-to-end RoBERTa + classification head_ | _(reference)_ | _69.3%_ | _0.689_ |

The embedding + XGBoost approach **actually exceeds** the end-to-end RoBERTa baseline (70.3% vs 69.3%). The vanilla models (no fine-tuning) still reach 58–62% — showing that even pre-trained BERT/RoBERTa encodings carry meaningful hate speech signal out of the box.

### Key Insights

1. **Embeddings are the bottleneck, not classifiers.** Switching from vanilla to fine-tuned embeddings gains ~8% F1. Switching classifiers within the same embeddings changes only ~2–4% F1. The embedding quality determines the ceiling.

2. **Pre-trained encoders already encode useful signal.** Vanilla BERT and RoBERTa — with zero hate speech training — reach 56–62% accuracy on 3-class classification. The pre-training on general text (BooksCorpus, Wikipedia, web crawl) gives these models enough semantic understanding to partially separate hate speech from normal content.

3. **Fine-tuning reshapes the geometry dramatically.** Silhouette scores jump from ≈0.00 (classes indistinguishable) to ≈0.18 (clear cluster structure). kNN purity rises from 41–69% to 72–86%. This geometric shift is what makes the shallow classifiers effective.

4. **Traditional ML can match end-to-end deep learning.** For binary: within 1.2%. For 3-class: actually surpasses it. The frozen encoder + XGBoost approach is faster to iterate, more modular, and requires no GPU for classifier training.

### Experiments at a Glance

| Section | What | Key Takeaway |
|---|---|---|
| §3 | Embedding geometry (silhouette, kNN, centroid sim, variance) | Fine-tuning creates real cluster structure; vanilla is flat |
| §4 | Vanilla embeddings + classifiers | 56–62% accuracy; pre-trained encoders still carry useful signal |
| §5 | Fine-tuned embeddings + classifiers | 83–86% binary, 66–70% 3-class; XGBoost wins consistently |
| §6 | Cost tradeoff analysis | Frozen embeddings + XGBoost matches end-to-end at fraction of cost |
| §7 | Classifier mechanics (LogReg, XGBoost, RF, MLP, PyTorch MLP) | How each classifier exploits the 768-dim space |
| §8 | Data science hygiene | No leakage; reproducible seeds; class weighting notes |
| §9 | Project status | All experiments complete; hack day plan in Section 10 |

---

## 1. Introduction

We investigated whether the geometry of transformer embeddings is a reliable predictor of downstream classifier performance — and whether a fine-tuned encoder's frozen embeddings can rival end-to-end fine-tuning.

All experiments use the **HateXplain** dataset (15,383 train / 1,924 test). We extracted 768-dimensional embeddings from five different encoders and measured both the geometric structure of those embeddings and the classification accuracy of shallow classifiers trained on them.

| Model | Type | Task |
|---|---|---|
| `roberta-base` | Vanilla (pre-trained only) | — |
| `bert-base-uncased` | Vanilla (pre-trained only) | — |
| `bert-binary-hatexplain-finetuned` | Fine-tuned | Hate vs non-hate (2-class) |
| `shareeep/bert-base-fine-tuned-hatexplain` | Fine-tuned | Hate / offensive / normal (3-class) |
| `shareeep/roberta-base-fine-tuned-hatexplain` | Fine-tuned | Hate / offensive / normal (3-class) |

All code and results are consolidated in `embedding-experiments.ipynb`.

---

## 2. Background

### BERT vs RoBERTa

Both models are transformer encoders with the same basic architecture (12 layers, 768 hidden dimensions, 110M parameters), but they were trained differently and those differences matter.

| Property | BERT | RoBERTa |
|---|---|---|
| Pre-training task | Masked language modelling (MLM) + Next sentence prediction (NSP) | MLM only — NSP was removed |
| Training data | 16GB (BooksCorpus + Wikipedia) | 160GB (same + web crawl data) |
| Masking | Static — mask decided once, reused | Dynamic — mask re-sampled each epoch |
| Tokeniser | WordPiece | Byte-Pair Encoding (BPE) |
| CLS token | `[CLS]` | `<s>` |
| Training time | Shorter | Much longer, larger batches |

The NSP task in BERT was meant to teach sentence relationships but was later found to contribute little. Removing it (RoBERTa) generally improves downstream performance. More training data and dynamic masking means RoBERTa captures more linguistic diversity — but after fine-tuning on the same task, both architectures perform comparably.

### How Embeddings Work

**Tokenisation** — raw text is split into subword tokens. BERT uses WordPiece, RoBERTa uses BPE. Both handle unknown words by decomposing them: "hateful" might become `["hate", "##ful"]`. A `[CLS]`/`<s>` token is prepended and a `[SEP]`/`</s>` token is appended.

**Token embeddings** — each token ID is looked up in a learned embedding table (token ID → 768-dim vector). These initial vectors encode identity but no context.

**Transformer layers** — the sequence of 768-dim vectors passes through 12 transformer layers. In each layer, every position attends to every other position via self-attention. After layer 12, each token's vector has seen the entire sequence multiple times.

**The CLS vector after layer 12** is not "the embedding of the word `[CLS]`" — it is a 768-dimensional summary built up from attending to all other tokens across all layers.

**Why 768 dimensions?** A design choice during pre-training. More dimensions = more capacity, but also more parameters and slower computation. 768 is standard for the "base" model size.

**What do the 768 numbers mean?** Individually, nothing interpretable — they are learned distributed representations. What matters is the geometry: two posts are "similar" if their 768-dim vectors point in a similar direction (high cosine similarity).

### The Classification Head

The encoder outputs a 768-dim vector. A **classification head** converts it to class probabilities:

```
[CLS] hidden state (768-dim)
        ↓
  Linear layer: 768 → num_classes   (a weight matrix + bias)
        ↓
  Softmax: raw scores → probabilities summing to 1
        ↓
  [p_hate, p_offensive, p_normal]   (for 3-class)
```

The head has almost no parameters — just `768 × num_classes` (≈2300 for 3-class). Almost all 110M parameters are in the encoder body. During fine-tuning, gradients flow through the entire encoder, reshaping its geometry to support the head's task. During embedding extraction, the head is discarded entirely.

### CLS Pooling

Both BERT and RoBERTa prepend a special token (`[CLS]`/`<s>`) whose job is to accumulate information across all 12 transformer layers. **CLS pooling** = take its final-layer hidden state as the embedding.

For fine-tuned classifiers this is especially meaningful — the training objective forces the CLS vector to be discriminative because the classification head sits directly on top of it.

For vanilla (pre-trained) models, the CLS token was never given a meaningful signal to aggregate sentence-level meaning. The geometry is anisotropic — it is no more useful than any other token.

### Rationale-Weighted Pooling

HateXplain includes **annotator rationales** — each human annotator highlighted specific words they considered evidence for their label. Instead of using only the CLS token, rationale-weighted pooling computes a **weighted average** of all content token embeddings, where each word's weight is its mean annotator rationale score (0.0 to 1.0).

The intuition: hate speech meaning is often carried by a small number of specific tokens. Upweighting those tokens should produce a more focused vector than the CLS summary.

Implementation details:
- Each whitespace word may map to multiple subword tokens; all subwords inherit the parent word's score
- Special tokens (`[CLS]`, `[SEP]`, etc.) always get weight 0
- For "normal" posts where no word was flagged (rationale all zeros), it falls back to plain mean pool

---

## 3. Experiment 1: Embedding Geometry

### Metrics

| Model | Pooling | Silhouette | kNN purity | Centroid sim | Within-class var |
|---|---|---|---|---|---|
| Vanilla RoBERTa | cls | −0.0028 | 0.4478 | 0.999963 | 0.000471 |
| Vanilla RoBERTa | rationale | +0.0125 | 0.5700 | 0.997843 | 0.012205 |
| Vanilla BERT | cls | −0.0044 | 0.4100 | 0.997052 | 0.051495 |
| Vanilla BERT | rationale | +0.0229 | 0.6886 | 0.956891 | 0.060711 |
| Fine-Tuned BERT Binary | cls | +0.1812 | 0.8270 | 0.572314 | 0.160208 |
| Fine-Tuned BERT Binary | rationale | +0.1683 | 0.8594 | 0.727932 | 0.097708 |
| Fine-Tuned BERT 3-Class | cls | +0.1554 | 0.7432 | 0.391726 | 0.154612 |
| Fine-Tuned RoBERTa 3-Class | cls | +0.1771 | 0.7210 | 0.636226 | 0.159959 |

**Metric key:** Silhouette ↑ better | kNN purity ↑ better | Centroid sim ↓ better | Within-class var ↓ better

### Interpretation

The results split cleanly into two groups:

**Vanilla models: near-zero or negative silhouette.** A negative silhouette score means the classes are worse than randomly arranged. This is the signature of **anisotropy**: all hidden states point in roughly the same direction regardless of content, so cosine similarity between any two posts is near 1.0. Class membership is invisible in the geometry.

**Fine-tuned models: substantial positive scores (0.15–0.18).** This is not marginal — it is a qualitative change. kNN purity confirms: among the 10 nearest neighbours of a fine-tuned embedding, 72–86% share the same label. For vanilla models that is only 41–69%.

**Why fine-tuning changes the geometry:** When a classification head is trained with cross-entropy loss, gradients flow backwards through the entire encoder, forcing it to rearrange all 768 dimensions to make the classification boundary achievable. A vanilla model has never seen a classification signal — its weights are optimised for masked language modelling.

**Scale of the gap:** Pre-trained silhouette ≈ 0.00 → fine-tuned silhouette ≈ 0.18. This is a ~50× improvement in cluster quality from fine-tuning alone — no architectural change, just supervised signal.

**Rationale pooling:** Improves vanilla baselines noticeably (0.02 vs −0.004 for BERT) but does not close the gap with fine-tuned models — confirming the bottleneck is the geometry of the weights, not the pooling readout.

---

## 4. Experiment 2: Vanilla Embeddings + Classifiers

### 3-Class (Vanilla RoBERTa)

| Classifier | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | **0.6216** | **0.6104** |
| MLP (256→128→64) | 0.6154 | 0.5937 |
| XGBoost | 0.6019 | 0.5696 |
| Random Forest | 0.5691 | 0.4959 |

### 3-Class (Vanilla BERT)

| Classifier | Accuracy | Macro F1 |
|---|---|---|
| MLP (256→128→64) | **0.5894** | **0.5672** |
| Logistic Regression | 0.5603 | 0.5551 |
| XGBoost | 0.5629 | 0.5248 |
| Random Forest | 0.5286 | 0.4603 |

The _normal_ class is systematically under-recalled (35–41%) across all vanilla models. Pre-trained embeddings conflate neutral and offensive content.

---

## 5. Experiment 3: Fine-Tuned Embeddings + Classifiers

### Binary task (hate vs non-hate)

| Embeddings | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned BERT Binary [rationale] | XGBoost | **0.8649** | **0.8403** |
| Fine-Tuned BERT Binary [rationale] | MLP (256→128→64) | 0.8571 | 0.8346 |
| Fine-Tuned BERT Binary [rationale] | Logistic Regression | 0.8467 | 0.8286 |
| Fine-Tuned BERT Binary [rationale] | Random Forest | 0.8560 | 0.8267 |
| Fine-Tuned BERT Binary [cls] | XGBoost | 0.8446 | 0.8172 |
| Fine-Tuned BERT Binary [cls] | MLP (256→128→64) | 0.8404 | 0.8159 |
| Fine-Tuned BERT Binary [cls] | Logistic Regression | 0.8290 | 0.8109 |
| Fine-Tuned BERT Binary [cls] | Random Forest | 0.8384 | 0.8057 |

Rationale pooling consistently outperforms CLS for the binary task. Random Forest slots in between MLP and LogReg.

### 3-class task (hate / offensive / normal)

| Embeddings | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned RoBERTa 3-Class [cls] | XGBoost | **0.7027** | **0.6907** |
| Fine-Tuned RoBERTa 3-Class [cls] | Random Forest | 0.6996 | 0.6857 |
| Fine-Tuned BERT 3-Class [cls] | XGBoost | 0.6949 | 0.6839 |
| Fine-Tuned RoBERTa 3-Class [cls] | Logistic Regression | 0.6881 | 0.6808 |
| Fine-Tuned RoBERTa 3-Class [cls] | MLP (256→128→64) | 0.6913 | 0.6758 |
| Fine-Tuned BERT 3-Class [cls] | Random Forest | 0.6866 | 0.6748 |
| Fine-Tuned BERT 3-Class [cls] | MLP (256→128→64) | 0.6757 | 0.6616 |
| Fine-Tuned BERT 3-Class [cls] | Logistic Regression | 0.6601 | 0.6531 |

Fine-tuned 3-class embeddings improve macro F1 by ~+0.08 over vanilla RoBERTa (0.61 → 0.69). RoBERTa slightly outperforms BERT across all classifiers. Random Forest ranks 2nd in the fine-tuned 3-class results, just behind XGBoost.

---

## 6. Combined Results & Cost Tradeoff

The geometry metrics correctly predicted the improvement from fine-tuned embeddings. The hypothesis — grounded in the silhouette jump from ≈0.00 to ≈0.18 — held:

| Approach | Training cost | Accuracy (binary) | Accuracy (3-class) |
|---|---|---|---|
| Vanilla embeddings + best classifier | Minimal (minutes) | ~56–63% | ~59–62% |
| Fine-tuned embeddings + XGBoost | Medium (encode once, train fast) | **86.5%** | **70.3%** |
| End-to-end fine-tuning (BERT / RoBERTa) | High (~1 hour on CPU) | **87.7%** | **69.3%** |

**The key insight: once you have a fine-tuned encoder, the classification head barely matters — the geometry does the work.** For binary, the gap between frozen embeddings + XGBoost and end-to-end fine-tuning is only 1.2%. For 3-class, frozen embeddings + XGBoost actually **exceeds** the end-to-end RoBERTa baseline (70.3% vs 69.3%) — the shallow classifier extracts slightly more signal from the frozen geometry than the classification head learned during fine-tuning.

### Two Ways to Use a Fine-Tuned Encoder

**End-to-end fine-tuning** (what `hate_speech_bert_binary.ipynb` does):

```
text → encoder (110M params, trainable) → CLS vector → classification head → loss
                      ↑
         gradients flow all the way back here
```

The head is just one linear layer, but gradients from the loss reshape the entire encoder.

**Frozen encoder + shallow classifier** (what `embedding-experiments.ipynb` does):

```
text → fine-tuned encoder (frozen) → CLS vector (fixed 768-dim) → XGBoost / LogReg / MLP
```

You use the encoder as a fixed feature extractor. This is more modular — once the encoder is saved, you can swap classifiers in minutes.

---

## 7. How the Classifiers Work

### Logistic Regression

Computes a weighted sum of all 768 features for each class, applies softmax, predicts the highest-probability class:

```
for each class k:  score_k = w_k · x + b_k    (768-dim dot product)
probabilities = softmax([score_0, score_1, ...])
```

A linear classifier — its performance tells you how linearly separable the embedding space is. `max_iter=1000` is important; the default 100 often terminates early on 768-dim inputs.

### XGBoost (Gradient-Boosted Trees)

Builds an ensemble of decision trees iteratively. Each tree splits on individual feature thresholds. Non-linear and handles feature interactions.

Settings: `n_estimators=200`, `learning_rate=0.05`, `max_depth=6` — appropriate for 768-dim input; deeper trees risk memorising noise dimensions.

### Random Forest

Ensemble of independent decision trees, each trained on a random subset of data and features. Each tree votes; majority wins. Unlike XGBoost, trees are independent — no sequential correction.

Settings: `n_estimators=200`, `max_depth=None` (no limit), `max_features='sqrt'` (each tree sees ~28 of 768 features per split), `class_weight='balanced'`.

Known limitation: with `max_depth=10` (too shallow), the _normal_ class recall can collapse to ~7%. Without a depth limit, the forest can carve out the fine-grained sub-regions needed for the intermediate offensive/normal boundary.

### MLP (Multi-Layer Perceptron)

The sklearn MLP (256→128→64, ReLU, `early_stopping=True`) applies non-linear transformations:

```
x (768) → Linear(768→256) → ReLU → ... → Linear(64→num_classes)
```

More representational power than LogReg, but more risk of overfitting. Early stopping with a 10% validation holdout mitigates this.

### PyTorch MLP — Explicit Layers & Activation Functions

The sklearn MLP hides the architecture behind a tuple. `embedding-experiments.ipynb` also includes a PyTorch implementation (`EmbeddingMLP`) where every layer, activation, and dropout is visible:

```
embedding (768) → Linear(768→256) → GELU → Dropout(0.1)
               → Linear(256→128) → GELU → Dropout(0.1)
               → Linear(128→64)  → GELU → Dropout(0.1)
               → Linear(64→num_classes)
```

This allows comparing activation functions — GELU (the activation used inside BERT/RoBERTa), ReLU, SiLU, and LeakyReLU — on the same architecture. GELU is the recommended choice since the input embeddings were shaped by GELU during pre-training.

### Why Do Classifiers Work At All If Silhouette ≈ 0?

- Silhouette ≈ 0 means the **global mean** pairwise structure is flat
- Classifiers find **local decision boundaries** that exploit subtle distributional differences
- Rationale pooling helps specifically by shifting the distribution in certain dimensions even when centroids barely move
- 56–63% accuracy is consistent with silhouette ≈ 0 — you can have non-trivial linear separability in high dimensions while the bulk of the mass overlaps

---

## 8. Data Science Hygiene

- **No data leakage:** HateXplain train/validation/test splits are the original author-defined splits. Embeddings are computed separately for each split.
- **Reproducibility:** all classifiers use `random_state=42`. XGBoost and sklearn MLP are deterministic given the seed.
- **No target leakage:** rationale scores are from the dataset annotations (ground truth), not model predictions.
- **Class imbalance handling:** the 3-class test set is roughly balanced (594/782/548), but offensive is over-represented. Logistic Regression and Random Forest use `class_weight='balanced'` to upweight the minority class. XGBoost and MLP do not use explicit class weighting, which partially explains the low _normal_ recall in 3-class results.
- **Frozen feature extractor:** encoder weights are frozen (`model.eval()`, `torch.no_grad()`). The classifier trains on extracted vectors only.

---

## 9. Project Status

### Completed

- [x] EDA and text length analysis (`hate-speech-eda-dap.ipynb`)
- [x] Vanilla embeddings + classifiers — binary and 3-class
- [x] BERT binary fine-tuned end-to-end — accuracy 87.7%, macro F1 0.874 (`hate_speech_bert_binary.ipynb`)
- [x] Embedding geometry comparison — silhouette, kNN purity, centroid similarity, within-class variance
- [x] t-SNE cluster analysis — vanilla, binary fine-tuned, 3-class fine-tuned (text-based metrics: 2D silhouette + centroid distance)
- [x] Pooling strategy comparison — CLS vs rationale-weighted
- [x] Shallow classifiers on fine-tuned embeddings — binary and 3-class (4 classifiers: LogReg, XGBoost, Random Forest, MLP)
- [x] Embedding arrays saved to disk (`embeddings/embedding_comparison_arrays.pkl`)
- [x] Consolidated notebook (`embedding-experiments.ipynb`) — all 30 cells verified in clean run
- [x] Findings document with full results (no placeholders)
- [x] PyTorch MLP with activation function comparison (GELU, ReLU, SiLU, LeakyReLU)
- [x] Random Forest added to classifier suite (4 classifiers total)
- [x] Superseded notebooks moved to `original/`

### Pending

- [ ] Class weighting in XGBoost/MLP classifiers — _normal_ recall (35%) remains a weak point in 3-class
- [ ] Cross-dataset evaluation — SGHateCheck and HateBenchSet loaders ready in notebook (commented out)
- [ ] Adversarial / DSPy experiments (`adversarial_hs_sg_plan.md`, `hate-speech-dspy.ipynb`) — separate track
- [ ] Hack day: localised hate speech generalisation (see Section 10)

---

## 10. Hack Day Plan: Localised Hate Speech Generalisation

### Starting Point

1. We have traditional ML pipelines leveraging rich text embeddings (via BERT, RoBERTa) with 4 traditional classifiers
2. Best embedding + classifier setup achieves **86.5% accuracy / 0.84 F1** (binary) and **70.3% / 0.69 F1** (3-class)
3. Raw encoders or larger LLMs (decoder-based) can give better results, but are much more expensive to run

### Goal

Evaluate how well these traditional ML pipelines **generalise to unseen data** — and specifically to **localised hate speech** that the models have never encountered.

### Phase 1: Baseline Generalisation (HateBenchSet)

Use HateBenchSet to get a baseline measure of how well the current models handle unseen English hate speech. This is a sanity check before introducing localisation — if the models already struggle on standard unseen data, that sets the floor.

### Phase 2: Localised Hate Speech (Singapore-Focused)

Go a step further with **completely new hate speech that is more likely to be unseen** — localised to Singapore, where slang, code-switching, and cultural references create distribution shift.

**Data generation approaches:**
- Use GenAI to generate localised examples of Singaporean hate speech
- Source from existing regional datasets or language data to understand SG slang better
- Create targeted test sets that stress-test the models on local vocabulary and phrasing

**Two paths for incorporating localised data:**

1. **LLM adapter path** — use localised data as fine-tuning data for LLM adapters (e.g. the existing MLX/LoRA pipeline)
2. **Embedding path** — fine-tune BERT/RoBERTa on the localised data to create embeddings that better understand Singaporean slang. Open question: this isn't a standard classification fine-tuning task — it's more about teaching the encoder to understand new vocabulary/registers. Possible approaches: continued MLM pre-training on SG text, or contrastive fine-tuning with SG examples

### Phase 3: Evaluation

From there, evaluate whether the updated models can generalise to the new task of classifying Singaporean-based hate speech:
- Compare vanilla → fine-tuned (HateXplain) → fine-tuned (HateXplain + SG) on the localised test set
- Measure if localised fine-tuning helps without degrading performance on the original HateXplain test set

### Viability Assessment

**Phase 1 is straightforward.** The HateBenchSet loader is already scaffolded in the notebook. The main work is wiring it into the existing `run_classifiers` pipeline, encoding the texts, and running inference. Half a day at most.

**Phase 2 has real substance but also real gaps:**

| Aspect | Status | Challenge |
|---|---|---|
| **SG test set creation** | Feasible | GenAI-generated hate speech needs careful human review — LLMs tend to produce sanitised or stereotypical examples. Real Singlish code-switches mid-sentence ("this kind of people damn jialat sia") and LLMs often get the register wrong. Manual curation or sourcing from real SG forums would be higher quality but slower. |
| **LLM adapter path** | Ready | The MLX/LoRA pipeline already exists (`hate-speech-llm-finetune-mlx.ipynb`). Adapters are saved. The question is whether a small SG dataset (likely <1k examples) is enough signal for LoRA to learn from — risk of overfitting to the style rather than the semantics. |
| **Embedding path (continued MLM)** | Viable but non-trivial | Continued MLM pre-training on SG text is the cleanest way to teach the tokenizer/encoder about Singlish without needing labels. HuggingFace makes this straightforward with `DataCollatorForLanguageModeling`. But: (1) you need a decent corpus of SG text (not just hate speech — general Singlish), (2) BERT's WordPiece tokenizer may struggle with novel SG tokens that get split into meaningless subwords, and (3) you need to balance continued MLM so it doesn't catastrophically forget English. |
| **Embedding path (contrastive)** | Harder | Requires labelled pairs ("these two SG texts are both hate") which brings you back to the data labelling problem. Skip unless you already have a labelled SG dataset. |
| **Evaluation fairness** | Needs thought | If you generate the test set with GenAI and also use GenAI for training data, you risk the model learning GenAI's style rather than real SG hate speech. The test set should ideally come from a different source than the training data. |

**Key gaps to address before the hack day:**

1. **Data source for Singlish corpus** — continued MLM needs general SG text, not just hate speech. Options: HardwareZone forums, Reddit r/singapore, NUS SMS Corpus, or SG government Hansard transcripts. Need to verify licensing/availability.
2. **Test set integrity** — decide upfront whether the SG hate test set is human-written, GenAI-generated, or sourced from existing datasets (e.g. portions of SGHateCheck). Mixing generation sources between train and test would invalidate the experiment.
3. **Binary vs 3-class** — the current pipeline supports both, but SG hate speech may not fit the hate/offensive/normal taxonomy cleanly. Singlish "vulgarities" (like "cb", "knn") are culturally normalised in casual speech but flagged as offensive by Western-trained models. May need to start with binary (hate vs non-hate) to avoid label ambiguity.
4. **Tokenizer coverage** — run a quick check on how many SG-specific tokens get split by BERT's WordPiece vs RoBERTa's BPE. If critical slang terms become 4–5 subword fragments, the embeddings lose meaning and continued MLM alone won't fix it (you'd need to extend the vocabulary, which is a bigger surgery).
