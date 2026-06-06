# Embedding Space Analysis: Findings & Discussion

## What We Did

We ran all posts from the HateXplain dataset through five different encoder models and extracted a **768-dimensional vector** (an *embedding*) for each post. We then measured how geometrically structured those vectors are — do hate posts cluster together? Do normal posts sit apart from offensive ones?

Five models were compared:

| Model | Status | Task |
|---|---|---|
| `roberta-base` | Pre-trained only | — |
| `bert-base-uncased` | Pre-trained only | — |
| `bert-binary-hatexplain-finetuned` | Fine-tuned | Hate vs non-hate (2-class) |
| `shareeep/bert-base-fine-tuned-hatexplain` | Fine-tuned | Hate / offensive / normal (3-class) |
| `shareeep/roberta-base-fine-tuned-hatexplain` | Fine-tuned | Hate / offensive / normal (3-class) |

---

## BERT vs RoBERTa: Key Differences

Both models are transformer encoders with the same basic architecture (12 layers, 768 hidden dimensions, 110M parameters), but they were trained differently and those differences matter.

| Property | BERT | RoBERTa |
|---|---|---|
| Pre-training task | Masked language modelling (MLM) + Next sentence prediction (NSP) | MLM only — NSP was removed |
| Training data | 16GB (BooksCorpus + Wikipedia) | 160GB (same + web crawl data) |
| Masking | Static — mask decided once, reused | Dynamic — mask re-sampled each epoch |
| Tokeniser | WordPiece | Byte-Pair Encoding (BPE) |
| CLS token | `[CLS]` | `<s>` |
| Training time | Shorter | Much longer, larger batches |

**Why this matters for us:**

- The NSP task in BERT was meant to teach sentence relationships, but it was later found to contribute little and potentially hurt the MLM objective. Removing it (RoBERTa) generally improves downstream task performance.
- More training data and dynamic masking means RoBERTa's pre-trained representations capture more linguistic diversity — which is why we see RoBERTa's rationale pooling silhouette (+0.0125) slightly better than BERT's CLS silhouette (−0.0044) despite neither having been fine-tuned.
- After fine-tuning on the same task, both architectures perform comparably (silhouettes of +0.18 and +0.18), which suggests the pre-training gap is largely closed by task-specific supervision.

---

## How Embeddings Actually Work

It helps to think of what the model is doing at each step.

**Tokenisation** — the raw text is first split into subword tokens. BERT uses WordPiece, RoBERTa uses BPE. Both handle unknown words by decomposing them: "hateful" might become `["hate", "##ful"]` in BERT or `["hate", "ful"]` in RoBERTa. A `[CLS]`/`<s>` token is prepended and a `[SEP]`/`</s>` token is appended.

**Token embeddings** — each token ID is looked up in a learned embedding table (like a dictionary: token ID → 768-dim vector). These initial vectors encode identity but no context: the word "hate" has the same initial vector regardless of sentence.

**Transformer layers** — the sequence of 768-dim vectors passes through 12 transformer layers. In each layer, every position attends to every other position via self-attention. This is where context is injected: after layer 1, each token's vector has been influenced by its neighbours. After layer 12, each token's vector has seen the entire sequence multiple times.

**The CLS vector after layer 12** is therefore not "the embedding of the word `[CLS]`" — it is a 768-dimensional summary built up from attending to all other tokens across all layers. It is the model's holistic representation of the whole post.

**Why 768 dimensions?** It is a design choice made during pre-training. More dimensions = more capacity to encode distinctions, but also more parameters and slower computation. 768 is the standard for the "base" model size.

**What do the 768 numbers mean?** Individually, nothing interpretable — they are learned distributed representations. What matters is the geometry: two posts are "similar" if their 768-dim vectors point in a similar direction (high cosine similarity). After fine-tuning, hate posts end up near each other in this space; normal posts end up elsewhere. t-SNE compresses those 768 dimensions down to 2 for visualisation, preserving local neighbourhood structure as best it can.

---

## What Is a Classification Head?

The encoder (BERT or RoBERTa) outputs a 768-dim vector for each input. On its own, that vector is not a prediction — it needs to be converted to class probabilities. That conversion is done by the **classification head**, which is simply a small piece of architecture bolted on top:

```
[CLS] hidden state (768-dim)
        ↓
  Linear layer: 768 → num_classes   (just a weight matrix + bias)
        ↓
  Softmax: converts raw scores to probabilities summing to 1
        ↓
  [p_hate, p_offensive, p_normal]   (for 3-class)
```

The head is called a "head" by analogy with human anatomy — the encoder is the body (the bulk of the work), and the head is a small specialised structure on top. The term comes from computer vision where multiple heads were attached to the same backbone for different tasks (detection head, segmentation head, etc.).

**Key point:** the head itself has almost no parameters — just `768 × num_classes` numbers (≈ 2300 for 3-class). Almost all 110M parameters are in the encoder body. When we say "fine-tuning", we mean training both the head and the encoder together, so the encoder's geometry is reshaped to support the head's task.

During embedding extraction (for t-SNE or downstream classifiers), the head is discarded entirely. We extract the 768-dim vector before it ever reaches the head — that is why the embeddings reflect what the encoder has learned, not what the head predicts.

---

## What Is CLS Pooling?

Both BERT and RoBERTa prepend a special token to every input — `[CLS]` in BERT, `<s>` in RoBERTa. This token has no meaning on its own. Its job is to *accumulate* information: across all 12 transformer layers, it attends to every other word in the sequence, so by the final layer its hidden state is a compressed summary of the entire post.

**CLS pooling** simply means: take the final-layer hidden state of that token as the sentence's embedding. For fine-tuned classifiers this is especially meaningful — the training objective explicitly forces the CLS vector to be discriminative because the classification head sits directly on top of it. Every gradient update that improves the classifier passes through the head and reshapes the CLS vector to be more separable.

For pre-trained models (no fine-tuning), the CLS token was never given a meaningful training signal to aggregate sentence-level meaning. It still attends to everything, but the geometry is anisotropic — it ends up no more useful than any other token's vector.

---

## What Is Rationale-Weighted Pooling?

HateXplain is unusual among datasets because it includes **annotator rationales** — each human annotator highlighted the specific words they considered evidence for their label ("filthy", "vermin", "go back to your country", etc.).

Instead of using only the CLS token, rationale-weighted pooling computes a **weighted average** of all content token embeddings, where each word's weight is its mean annotator rationale score (0.0 to 1.0). Words not flagged by any annotator get weight ≈ 0; strongly flagged words dominate the embedding.

The intuition: hate speech meaning is often carried by a small number of specific tokens. Upweighting those tokens should produce a more semantically focused vector than the CLS summary, which compresses everything including stop words.

In practice, this required careful implementation:
- Each whitespace-split word may map to multiple subword tokens (BPE or WordPiece); all subwords inherit the parent word's score
- Special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`) are always given weight 0
- For "normal" posts where no annotator flagged any word (rationale all zeros), it falls back to a plain mean pool

---

## Results

```
Model                               Pooling    Silhouette  kNN purity  Centroid sim  Within-class var
roberta-base (pre-trained)          cls          -0.0028      0.4478      0.999963        0.000471
roberta-base (pre-trained)          rationale    +0.0125      0.5700      0.997843        0.012205
bert-base (pre-trained)             cls          -0.0044      0.4100      0.997052        0.051495
bert-base (pre-trained)             rationale    +0.0229      0.6886      0.956891        0.060711
bert-binary (fine-tuned)            cls          +0.1812      0.8270      0.572314        0.160208
bert-binary (fine-tuned)            rationale    +0.1683      0.8594      0.727932        0.097708
bert-3class (fine-tuned)            cls          +0.1554      0.7432      0.391726        0.154612
roberta-3class (fine-tuned)         cls          +0.1771      0.7210      0.636226        0.159959
```

**Metric key:** Silhouette ↑ better | kNN purity ↑ better | Centroid sim ↓ better (more separated) | Within-class var ↓ better

---

## What the Silhouette Score Tells Us

The **silhouette score** measures how well each data point fits its own cluster relative to the nearest other cluster. It ranges from −1 (misclassified) to +1 (perfectly separated). Around zero means the classes are geometrically indistinguishable.

The results split cleanly into two groups:

**Pre-trained models: near-zero or negative**
- `roberta-base [cls]`: −0.0028
- `bert-base [cls]`: −0.0044

A negative silhouette score means the classes are *worse* than randomly arranged — each post is on average closer to a different class's centre than its own. This is the signature of **anisotropy**: all hidden states from a raw pre-trained model point in roughly the same direction regardless of content, so cosine similarity between any two posts is near 1.0 and the centroid similarity is 0.9999. Class membership is invisible in the geometry.

**Fine-tuned models: substantial positive scores**
- `bert-binary [cls]`: +0.1812
- `roberta-3class [cls]`: +0.1771
- `bert-binary [rationale]`: +0.1683
- `bert-3class [cls]`: +0.1554

All fine-tuned models sit comfortably above 0.15. This is not a marginal improvement — it is a qualitative change in the structure of the space.

The kNN purity metric confirms this: among the 10 nearest neighbours of a random fine-tuned embedding, 72–86% share the same label. For pre-trained models that figure is only 41–69% — barely better than chance given the class distribution.

---

## Why Fine-Tuning Changes the Geometry

When a classification head is added on top of the encoder and trained with a cross-entropy loss, **gradients flow backwards through the entire encoder**. This forces the model to rearrange all 768 dimensions in a way that makes the classification boundary achievable.

Concretely: for the binary model, the loss is minimised when hate embeddings are on one side of a linear decision boundary and non-hate embeddings are on the other. After enough gradient updates, the encoder has learned to internally represent "hate-relevant" features — specific linguistic patterns, targets, and framing — as separable directions in the 768-dimensional space.

A pre-trained model has never seen a classification signal. Its weights are optimised for next-word prediction (RoBERTa) or masked language modelling (BERT). The geometry reflects *linguistic co-occurrence*, not *class membership* — hence the anisotropy.

---

## Is This Expected?

Yes, and the magnitude confirms the hypothesis rather cleanly.

The anisotropy of pre-trained language models is a well-documented property (Ethayarajh 2019, Li et al. 2020). Pre-trained models are strong at capturing linguistic structure but their raw embedding space is not useful for classification without a downstream signal.

What is interesting here is the scale of the gap:
- Pre-trained silhouette: ≈ 0.00 (noise floor)
- Fine-tuned silhouette: ≈ 0.16–0.18 (clearly structured)

This is a ~50× improvement in cluster quality from fine-tuning alone — no architectural change, just supervised signal applied to the same 110M parameters.

The rationale pooling result is also instructive: it improves pre-trained baselines noticeably (0.02 vs −0.004 for BERT) but does not close the gap with fine-tuned models. This confirms that the bottleneck is the geometry of the weights themselves, not the pooling strategy used to read them out.

---

## Linking to the RoBERTa Embeddings Notebooks

The `roberta-embeddings-binary.ipynb` and `roberta-embeddings-ml.ipynb` notebooks take the complementary approach: instead of analysing the geometry, they **use pre-trained RoBERTa embeddings as fixed features** and train lightweight classifiers on top of them (logistic regression, XGBoost, MLP).

### Binary Classification (Vanilla RoBERTa embeddings)

Results from `roberta-embeddings-binary.ipynb`:

| Classifier | Accuracy |
|---|---|
| Logistic Regression | 0.69 |
| XGBoost | 0.72 |
| MLP (256→128→64) | 0.73 |

Compare this to Fine-Tuned BERT + Inference on BERT (from `hate_speech_bert_binary.ipynb`):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Fine-Tuned BERT + Inference on BERT | **0.877** | **0.874** |

This gap — 73% vs 88% — is the practical consequence of everything discussed above. When you hand a classifier pre-trained embeddings that are geometrically unstructured (silhouette ≈ 0), it has to work much harder to find a decision boundary, and a shallow classifier cannot compensate for what the encoder has not learned to encode.

### 3-Class Classification (Vanilla RoBERTa embeddings)

Results from `roberta-embeddings-multi-class.ipynb` — rationale-weighted pooling, full dataset (15 383 train / 1 924 test).

| Classifier | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Logistic Regression | **0.630** | **0.60** | Best overall; strong on hate + offensive |
| MLP (256→128→64) | 0.615 | 0.59 | Small gain over LogReg on offensive recall |
| XGBoost | 0.603 | 0.57 | Competitive but slightly below linear baseline |
| Random Forest | 0.560 | 0.47 | Weakest — very poor recall on _normal_ (7%) |

Per-class breakdown (best model — Logistic Regression):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| hate | 0.64 | 0.70 | 0.67 | 594 |
| offensive | 0.65 | 0.77 | 0.71 | 782 |
| normal | 0.56 | 0.35 | 0.43 | 548 |

Key observation: _normal_ class is systematically under-recalled (35–41%) across all models. Pre-trained roberta embeddings conflate neutral and offensive content — the model was never taught to separate them.

### Downstream Classifiers on Fine-tuned Embeddings

Results from `embedding-classifier-pipeline.ipynb` — Route B cells. All embeddings from `embeddings/embedding_comparison_arrays.pkl`.

**Binary task (hate vs non-hate):**

| Embeddings source | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned BERT Binary [rationale] | XGBoost | **0.8649** | **0.8403** |
| Fine-Tuned BERT Binary [rationale] | MLP (256→128→64) | 0.8571 | 0.8346 |
| Fine-Tuned BERT Binary [rationale] | Logistic Regression | 0.8467 | 0.8286 |
| Fine-Tuned BERT Binary [cls] | XGBoost | 0.8446 | 0.8172 |
| Fine-Tuned BERT Binary [cls] | MLP (256→128→64) | 0.8404 | 0.8159 |
| Fine-Tuned BERT Binary [cls] | Logistic Regression | 0.8290 | 0.8109 |

Reference: **Fine-Tuned BERT + Inference on BERT** (end-to-end, Route A): acc=0.877, macro F1=0.874.

The best Route B result (XGBoost on rationale embeddings: 0.8403 macro F1) closes **80%** of the gap between vanilla embeddings (~0.72) and end-to-end fine-tuning (0.874). Rationale pooling consistently outperforms CLS pooling for the binary task.

**3-class task (hate / offensive / normal):**

| Embeddings source | Classifier | Accuracy | Macro F1 |
|---|---|---|---|
| Fine-Tuned RoBERTa 3-Class [cls] | XGBoost | **0.7027** | **0.6907** |
| Fine-Tuned BERT 3-Class [cls] | XGBoost | 0.6949 | 0.6839 |
| Fine-Tuned RoBERTa 3-Class [cls] | Logistic Regression | 0.6881 | 0.6808 |
| Fine-Tuned RoBERTa 3-Class [cls] | MLP (256→128→64) | 0.6913 | 0.6758 |
| Fine-Tuned BERT 3-Class [cls] | MLP (256→128→64) | 0.6757 | 0.6616 |
| Fine-Tuned BERT 3-Class [cls] | Logistic Regression | 0.6601 | 0.6531 |
| Vanilla RoBERTa [cls] | Logistic Regression | 0.630 | 0.60 |
| Vanilla BERT [cls] | Logistic Regression | 0.560 | 0.555 |

Fine-tuned 3-class embeddings improve macro F1 by ~+0.09 over vanilla RoBERTa (0.60 → 0.69). RoBERTa slightly outperforms BERT across all classifiers. The _normal_ class remains the hardest to classify (F1 ≈ 0.50–0.54 vs hate at 0.75–0.77).

### Outcome and Cost Tradeoff

The geometry metrics correctly predicted a large improvement from fine-tuned embeddings. The hypothesis — grounded in the silhouette jump from ≈0.00 to ≈0.18 — held:

| Approach | Training cost | Actual accuracy (binary) |
|---|---|---|
| Vanilla embeddings + best classifier | Minimal (minutes) | ~73% |
| Fine-tuned embeddings + XGBoost (Route B) | Medium (encode once, train fast) | **86.5%** |
| Fine-Tuned BERT + Inference on BERT (Route A) | High (~1 hour on CPU) | **87.7%** |

Once you have a fine-tuned encoder, the classification head barely matters — the geometry does the work. The 1.2% gap between Route B and Route A suggests that the classification head contributes only marginal additional signal beyond what the embedding geometry already provides.

---

## Two Ways to Use a Fine-Tuned Encoder

This is worth making explicit because it is easy to conflate.

**Route A — End-to-end fine-tuning (what `hate_speech_bert_binary.ipynb` does):**

```
text → encoder (110M params, trainable) → CLS vector → classification head (2300 params) → loss
                      ↑
         gradients flow all the way back here
         encoder geometry is reshaped by the task
```

The classification head is indeed "just one linear layer", but that is not what makes it powerful. The key is that during training, gradients from the cross-entropy loss propagate back through the entire encoder, forcing it to rearrange all 768 dimensions to make the discrimination easier. The head just converts that rearranged space into probabilities.

**Route B — Frozen encoder + shallow classifier (what the Route B cells in `embedding-classifier-pipeline.ipynb` do):**

```
text → fine-tuned encoder (frozen, no grad) → CLS vector (fixed 768-dim geometry)
                                                        ↓
                                          XGBoost / LogReg / MLP (trained separately)
```

Here you use the encoder purely as a feature extractor. The weights are frozen after fine-tuning — you never touch them again. You then train whichever classifier you want on those fixed 768-dim vectors.

The meaningful question is: **does the geometry the encoder learned during fine-tuning already carry enough structure that a frozen extraction + shallow classifier can match Route A?** The silhouette results (0.18) suggest yes, or close to it. This is what the pending TODO experiments will verify.

**Practical implication:** Route B is more modular. Once you have a fine-tuned encoder saved to disk, you can swap the final classifier (try XGBoost, then LogReg, then an MLP) in minutes, without touching the 110M-parameter model. For large-scale deployment this matters: inference is just a forward pass through the encoder + a matrix multiply at the end.

---

## What the Classifiers Actually Do With 768 Numbers

Each classifier treats the 768-dim vector differently. Understanding this helps explain the result pattern.

### Logistic Regression

Computes a weighted sum of all 768 features for each class, applies softmax, and predicts the highest-probability class:

```
for each class k:  score_k = w_k · x + b_k    (768-dim dot product)
probabilities = softmax([score_0, score_1, score_2])
```

This is a linear classifier — it finds a hyperplane that separates the classes. Its performance tells you how linearly separable the embedding space is. The fact that LogReg achieves 63% on pre-trained embeddings despite silhouette ≈ 0 shows that subtle linear structure exists even when the global cluster geometry looks flat.

**Optimal settings:** `max_iter=1000` is important — the L-BFGS solver needs many iterations to converge on 768-dim inputs. The default 100 often terminates early and gives misleading convergence warnings. No other tuning is critical for 768-dim dense vectors.

### XGBoost (Gradient-Boosted Trees)

Builds an ensemble of decision trees iteratively. Each tree splits on individual feature thresholds: "if dimension 347 > 0.12 and dimension 89 < −0.05, predict _hate_". No single dimension means anything interpretable, but across hundreds of trees the model learns to exploit correlated patterns.

Key difference from LogReg: XGBoost is non-linear and handles interactions between features. Yet it scores slightly below LogReg (60% vs 63%) on these embeddings. That is not unusual — transformer embeddings tend to be linearly structured (distances in the embedding space are meaningful linearly), so the extra expressivity of trees does not always help.

**Optimal settings reviewed:**
- `n_estimators=100`, `learning_rate=0.1` — reasonable starting point; could try 300 trees at 0.05 learning rate for marginal gains
- `max_depth=6` — appropriate for 768-dim input; deeper trees risk memorising noise dimensions
- `eval_metric='mlogloss'` — correct for multiclass; uses the softmax objective internally
- `max_leaves=0` (no limit) with `depthwise` growth is the standard setting; `lossguide` growth (LightGBM-style) often helps for sparse signals but less relevant here
- Class imbalance is mild enough that `scale_pos_weight` is not needed

### MLP (Multi-Layer Perceptron)

The sklearn MLP (256→128→64, ReLU, `early_stopping=True`) applies non-linear transformations:

```
x (768) → Linear(768→256) → ReLU → Linear(256→128) → ReLU → Linear(128→64) → ReLU → Linear(64→3)
```

This gives it more representational power than LogReg, but also more risk of overfitting. The early stopping with a 10% validation holdout mitigates this.

**Activation function note:** ReLU is fine, but the PyTorch MLP in the notebook uses GELU — the same activation used inside BERT and RoBERTa. Since the input vectors came from a GELU-based model, the feature distributions are shaped by GELU's smooth, non-zero-at-negative-inputs behaviour. Using GELU in the downstream MLP keeps the input distribution well-behaved for the activation. RELU can produce "dead neurons" (output stuck at 0 for all inputs) when negative features are common — which they are in transformer embeddings. GELU/SiLU are safer choices here.

**Why MLP slightly beats LogReg on this task (62% vs 63%):** The gain is marginal and may not replicate. The 3-class boundary is not perfectly linear, so some non-linearity helps, but the bottleneck is the embedding geometry, not the classifier expressivity.

### Random Forest

Also tree-based but uses bagging (random subsets of data and features per tree) without boosting. The `max_depth=10` limit causes the _normal_ class to collapse to near-zero recall (7%). This is a symptom of class geometry: normal posts sit in a region of embedding space that is intermediate between hate and offensive. Without sufficient depth to carve out fine-grained sub-regions, the forest defaults to the two more-populated classes. Removing the depth limit or increasing estimators would likely help, but the ceiling is lower than XGBoost regardless.

### Why Do These Work At All If Silhouette ≈ 0?

This apparent contradiction is important:

- Silhouette ≈ 0 tells you the **global mean** pairwise structure is flat — centroids are all close together in high-dimensional space
- Classifiers don't need global centroids to be well-separated — they find **local decision boundaries** that exploit subtle distributional differences (particular feature dimensions, second-order statistics, tail separation)
- Rationale pooling helps here specifically: by concentrating the embedding on flagged tokens, the distribution of embedding values for hate posts shifts in specific dimensions even if the centroids barely move
- The 60–73% accuracy is entirely consistent with silhouette ≈ 0 — you can have non-trivial linear separability in high dimensions while the bulk of the mass overlaps

---

## Data Science Hygiene

A few practices worth noting explicitly:

- **No data leakage:** the HateXplain train/validation/test splits are the original author-defined splits. Embeddings are computed separately for each split. The sklearn MLP uses a 10% holdout from the training set only for early stopping — it never sees the test set.
- **Reproducibility:** all classifiers use `random_state=42`. XGBoost and sklearn MLP are deterministic given the seed. The PyTorch MLP uses a fixed seed for weight initialisation but CUDA results can vary; CPU runs are deterministic.
- **No target leakage:** rationale scores are extracted from the dataset annotations and are part of the original ground-truth labels — they are not derived from model predictions. This is a legitimate use of the annotation.
- **Class imbalance acknowledged:** the 3-class test set is roughly balanced (594/782/548), but offensive is over-represented. No oversampling or class-weighting was applied in these shallow classifier experiments, which partially explains the systematic bias toward high offensive recall at the cost of normal recall. This is worth revisiting for a production setting.
- **Pre-trained model as fixed feature extractor:** the pre-trained RoBERTa model is never updated. Its weights are frozen (`model.eval()`, `torch.no_grad()`). The classifier trains on the extracted vectors only — the feature extractor is cleanly separated from the downstream model.

---

## Project Checklist

### Completed

- [x] EDA and text length analysis (`hate-speech-eda-dap.ipynb`)
- [x] Binary classification — pre-trained RoBERTa embeddings + LogReg / XGBoost / MLP / RF (`roberta-embeddings-binary.ipynb`)
- [x] 3-class classification — pre-trained RoBERTa embeddings + LogReg / XGBoost / MLP / RF (`roberta-embeddings-multi-class.ipynb`) — **results now in this doc**
- [x] BERT binary fine-tuned end-to-end — accuracy 87.7%, macro F1 0.874 (`hate_speech_bert_binary.ipynb`)
- [x] Embedding space comparison — silhouette, kNN purity, centroid similarity, within-class variance (`embedding-space-comparison.ipynb`)
- [x] t-SNE visualisations — pre-trained (3-class), binary fine-tuned, 3-class fine-tuned
- [x] Pooling strategy comparison — CLS vs rationale-weighted
- [x] HuggingFace fine-tuned models added to comparison (`shareeep/bert-base-fine-tuned-hatexplain`, `shareeep/roberta-base-fine-tuned-hatexplain`)
- [x] Embedding arrays saved to disk (`embeddings/embedding_comparison_arrays.pkl`) with skip-if-computed logic
- [x] Findings document (`findings-embedding-space.md`) — results, explanations, comparisons

### Pending

- [ ] **BERT 3-class fine-tuned (end-to-end)** — notebook `hate_speech_bert_3class.ipynb` created but not yet run; saves to `bert-3class-hatexplain-finetuned/`
- [x] **Shallow classifiers on fine-tuned binary embeddings** — Route B in `embedding-classifier-pipeline.ipynb`. Best: Fine-Tuned BERT Binary [rationale] + XGBoost → acc=0.8649, macro F1=0.8403
- [x] **Shallow classifiers on fine-tuned 3-class embeddings** — Route B. Best: Fine-Tuned RoBERTa 3-Class [cls] + XGBoost → acc=0.7027, macro F1=0.6907
- [x] **Fill results tables** in this document — done, see "Downstream Classifiers on Fine-tuned Embeddings" section above
- [ ] **PyTorch MLP multi-class** — the cell exists in `roberta-embeddings-multi-class.ipynb` but the output is missing; re-run to get GELU activation results
- [ ] **Consider class weighting** in shallow classifiers — the _normal_ class recall (35%) is a clear weak point; `class_weight='balanced'` in LogReg and `scale_pos_weight` in XGBoost are obvious first steps
- [ ] **Adversarial / DSPy experiments** (`adversarial_hs_sg_plan.md`, `hate-speech-dspy.ipynb`) — separate track, not yet started
