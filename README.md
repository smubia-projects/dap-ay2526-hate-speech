# 11th-DAP-hate-speech

Hate speech detection project for the 11th DAP Project under SMUBIA. Covers BERT/RoBERTa fine-tuning, embedding-based classifiers, LLM zero-shot/few-shot, and synthetic data generation for Singapore-specific hate speech.

## Repo Structure

```
11th-DAP-hate-speech/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── notebooks/                          # All Jupyter notebooks, grouped by topic
│   ├── bert/
│   │   └── hate_speech_bert_binary.ipynb
│   ├── roberta/
│   │   ├── hatespeech_keith_roberta_finetune.ipynb
│   │   ├── roberta_inference.ipynb
│   │   └── roberta_sg_experiments.ipynb
│   ├── embeddings/
│   │   └── embedding-experiments.ipynb
│   ├── llm/
│   │   ├── hate-speech-dspy.ipynb
│   │   ├── hate-speech-llm-finetune-mlx.ipynb
│   │   ├── hate-speech-llm-local.ipynb
│   │   └── hate-speech-llm-openrouter.ipynb
│   ├── synthetic/
│   │   ├── synthetic_generation_colab_gemini_final.ipynb   # Original Gemini generation (Colab)
│   │   ├── synthetic_judge_validation.ipynb                # LLM-as-a-judge quality validation
│   │   ├── synthetic_dataset_repair_policy.ipynb           # Cleanup, dedup, and term policy
│   │   └── synthetic_dataset_regeneration.ipynb            # Targeted regeneration of removed rows
│   └── archive/                        # Old/superseded notebooks
│       ├── [old]hate-speech-llm-colab.ipynb
│       ├── dap-hatebenchset-eda.ipynb
│       ├── embedding-classifier-pipeline.ipynb
│       ├── embedding-space-comparison.ipynb
│       ├── hate_speech_bert.ipynb
│       ├── hate-speech-bert.ipynb
│       ├── hate-speech-eda-dap.ipynb
│       ├── roberta-embeddings-analysis.ipynb
│       ├── roberta-embeddings-binary.ipynb
│       └── roberta-embeddings-multi-class.ipynb
│
├── data/                               # All datasets
│   ├── hatexplain/                     # HateXplain (15,383 train / 1,924 test)
│   │   ├── hatexplain_train.jsonl
│   │   ├── hatexplain_validation.jsonl
│   │   └── hatexplain_test.jsonl
│   ├── hatebenchset/                   # HateBenchSet
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   ├── test.jsonl
│   │   └── data_list.md
│   ├── singlish/                       # Singapore-specific data
│   │   ├── raw/                        #   Scraped Singlish terms
│   │   │   ├── singlish_parsed.jsonl
│   │   │   └── singlish_raw.jsonl
│   │   └── synthetic/                  #   Generated SG hate speech
│   │       ├── 20-terms/               #     Balanced 20-term dataset
│   │       │   ├── chosen-20-terms.json
│   │       │   ├── train.jsonl
│   │       │   ├── val.jsonl
│   │       │   └── test.jsonl
│   │       └── original-incomplete/    #     Earlier incomplete run
│   │           repair_outputs/         #     Post-cleanup artifacts
│   │           ├── cleaned_splits/     #       Deduplicated base splits
│   │           ├── regenerated_outputs/
│   │           │   ├── regenerated_rows.jsonl
│   │           │   └── repaired_splits/  # ← final repaired dataset (train/val/test)
│   │           ├── delete_candidates.json
│   │           ├── removed_hate_blueprints.json
│   │           ├── term_policy.json
│   │           └── post_cleanup_summary.json
│   └── embeddings/                     # Pre-computed embeddings (.pkl) & t-SNE plots (.png)
│
├── models/                             # Trained model artifacts (gitignored)
│   ├── bert-binary-hatexplain-finetuned/
│   ├── adapters/
│   │   └── Qwen2.5-3B-Instruct-4bit-hatexplain/   # LoRA adapters
│   └── runs/                           # Training checkpoints
│       ├── bert-binary-hatexplain/
│       └── bert-3class-hatexplain/
│
├── results/                            # Evaluation outputs
│   ├── model_scores.csv
│   └── hate_speech_predictions_openrouter.csv
│
├── scripts/                            # Python scripts
│   ├── predictions_app.py              #   Streamlit demo app
│   ├── view_dataset.py                 #   Streamlit dataset viewer (with judge flags + manual edit)
│   ├── singlish_scraper.py             #   Web scraper for Singlish terms
│   ├── parse_raw.py                    #   Parse raw scraped data
│   ├── generate_synthetic.py           #   Generate synthetic training data
│   ├── synthetic_generation.py         #   Gemini API generation logic
│   └── synthetic_processing.py         #   Post-processing for synthetic data
│
└── docs/                               # Documentation
    ├── presentation-slides.md
    ├── findings-embedding-space.md
    ├── findings-embedding-space-old.md
    ├── findings-roberta-finetune.md
    ├── adversarial_hs_sg_plan.md
    ├── dspy_explainer.md
    ├── hate_speech_vs_offensive_policies.md
    ├── synthetic_dataset_repair_decisions.md  # Decision log for dataset repair & regeneration
    ├── to-do.md
    ├── IS469_Shariff_Indiv_Presentation_Slides.pdf
    └── IS469_Shariff_Indiv_Report.pdf
```

## Quick Start

```bash
# 1. Pin to Python 3.13 and install dependencies
echo "3.13" > .python-version
uv venv --python 3.13
uv sync

# 2. Run the predictions demo app
uv run streamlit run scripts/predictions_app.py

# 3. Run the synthetic dataset viewer
uv run streamlit run scripts/view_dataset.py
```

## Breaking Changes (Repo Reorganisation)

The repo was restructured on 14 Mar 2026. If you have existing work that references old paths, here's what moved:

| Old path | New path |
|----------|----------|
| `hatexplain/` | `data/hatexplain/` |
| `hatebenchset/` | `data/hatebenchset/` |
| `singlish-data/singlish/` | `data/singlish/raw/` |
| `singlish-data/gemini-hate-speech-20-terms/synthetic_training_20_terms_simplfied.jsonl` | `data/singlish/synthetic/20-terms/train.jsonl` |
| `singlish-data/gemini-hate-speech-20-terms/synthetic_val_20_terms_simplfied.jsonl` | `data/singlish/synthetic/20-terms/val.jsonl` |
| `singlish-data/gemini-hate-speech-20-terms/synthetic_test_20_terms_simplfied.jsonl` | `data/singlish/synthetic/20-terms/test.jsonl` |
| `embeddings/` | `data/embeddings/` |
| `bert-binary-hatexplain-finetuned/` | `models/bert-binary-hatexplain-finetuned/` |
| `adapters/` | `models/adapters/` |
| `runs/` | `models/runs/` |
| `scraping/*.py` | `scripts/*.py` |
| `predictions_app.py` | `scripts/predictions_app.py` |
| `llm/*.ipynb` | `notebooks/llm/*.ipynb` |
| `original/*.ipynb` | `notebooks/archive/*.ipynb` |
| Root-level notebooks (e.g. `embedding-experiments.ipynb`) | `notebooks/<topic>/` |
| Root-level docs (e.g. `findings-*.md`, `to-do.md`) | `docs/` |

**Notebooks that load data with relative paths will need updating.** Key paths to fix:

- `hatexplain/` → `data/hatexplain/` (used in embedding-experiments, bert, roberta notebooks)
- `singlish-data/` → `data/singlish/` (used in roberta_sg_experiments)
- `bert-binary-hatexplain-finetuned/` → `models/bert-binary-hatexplain-finetuned/` (used in embedding-experiments)
- `runs/` → `models/runs/` (used in bert training notebooks)

The Colab notebooks (`roberta_sg_experiments`, `synthetic_generation_colab_gemini_final`) use Google Drive paths and are **not affected** by this reorganisation.