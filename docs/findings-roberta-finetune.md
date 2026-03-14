# RoBERTa Fine-Tuning for 3-Class Hate Speech Classification

## Setup

- **Model**: `cardiffnlp/twitter-roberta-base-hate` (pre-trained on Twitter hate speech, originally 2-class)
- **Task**: 3-class classification — hatespeech, offensive, normal
- **Dataset**: HateXplain (majority-vote labels)
- **Environment**: Google Colab, Tesla T4 GPU

The original model's 2-class head was replaced with a fresh 3-class head. All transformer layers were fine-tuned end-to-end.

## Data

| Split | Count | Hatespeech | Offensive | Normal |
|-------|------:|:----------:|:---------:|:------:|
| Train | 15,383 | 30.9% | 40.6% | 28.5% |
| Val   |  1,922 | 30.9% | 40.6% | 28.5% |
| Test  |  1,924 | 30.9% | 40.6% | 28.5% |

Class weights were applied to the loss function to handle the slight imbalance: hatespeech 1.08, offensive 0.82, normal 1.17.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Batch size     | 16    |
| Epochs         | 5     |
| Learning rate  | 2e-5  |
| Warmup ratio   | 10%   |
| Weight decay   | 0.01  |
| Max length     | 128   |
| Optimizer      | AdamW |

Best checkpoint was saved at **epoch 2** (val macro F1 = **0.6987**). After that, validation F1 plateaued/declined — the model began overfitting.

| Epoch | Train Loss | Val Macro F1 |
|------:|-----------:|-------------:|
| 1     | 0.828      | 0.683        |
| 2     | 0.647      | **0.699**    |
| 3     | 0.509      | 0.698        |
| 4     | 0.370      | 0.696        |
| 5     | 0.280      | 0.681        |

## Test Results (Best Checkpoint — Epoch 2)

| Class      | Precision | Recall | F1   | Support |
|------------|----------:|-------:|-----:|--------:|
| Hatespeech | 0.72      | 0.82   | 0.77 | 594     |
| Offensive  | 0.74      | 0.77   | 0.76 | 782     |
| Normal     | 0.58      | 0.45   | 0.51 | 548     |
| **Macro avg** | **0.68** | **0.68** | **0.68** | 1,924 |

- **Accuracy**: 70%
- **Macro F1**: 0.6774

The "normal" class is the weak spot — the model tends to misclassify normal text as offensive or hatespeech (recall only 45%). Hatespeech and offensive are reasonably well separated from each other.

## Example Predictions

| Input | Predicted | Confidence | Correct? |
|-------|-----------|:----------:|:--------:|
| "I hate all people from that country, they should go back where they came from." | HATESPEECH | 79.9% | ✓ |
| "These people are ruining our country and should be deported." | HATESPEECH | 69.7% | ✓ |
| "We need to get rid of every single Chinese in this neighborhood right now." | HATESPEECH | 66.9% | ✓ |
| "If you are a Jew, you deserve to be beaten up. Simple as that." | HATESPEECH | 58.9% | ✓ |
| "The new update for this app is completely broken. Fix it!!" | OFFENSIVE | 87.6% | ✗ |
| "This is a great movie, I really enjoyed it." | OFFENSIVE | 92.8% | ✗ |
| "Just tried the new bakery on 5th street. The croissants are amazing!" | OFFENSIVE | 92.0% | ✗ |
| "I am so tired of this rain. It's been gloomy for a straight week." | OFFENSIVE | 84.7% | ✗ |
| "You are literally the dumbest person on the internet. Delete your account" | NORMAL | 68.2% | ✗ |
| "Shut up and sit down. Your opinions are completely worthless." | NORMAL | 63.8% | ✗ |

The model catches real hate speech, but **clearly benign text gets labelled offensive with high confidence**, and **genuinely offensive text gets labelled normal**. This is a significant calibration problem — the model has a strong bias toward the "offensive" class (the majority class in training).

## Key Takeaways

1. **0.68 macro F1** on HateXplain 3-class is a reasonable baseline, consistent with our earlier embedding + traditional ML results (frozen RoBERTa embeddings + XGBoost got 0.703)
2. **Normal class is undertrained** — 45% recall means over half of normal posts get flagged as harmful. This is the same weakness seen in the embedding experiments
3. **Offensive bias** — the model defaults to "offensive" for ambiguous or unseen text, likely because it's the largest class (40.6%)
4. **Early stopping matters** — best performance at epoch 2, then gradual overfit. A proper early stopping callback would have saved compute
5. **End-to-end fine-tuning didn't outperform frozen embeddings + XGBoost** — this is notable. The traditional ML pipeline (70.3% accuracy) actually edges out 5 epochs of end-to-end fine-tuning (70% accuracy) while being much cheaper to iterate on
