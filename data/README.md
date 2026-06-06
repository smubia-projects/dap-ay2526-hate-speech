# Singlish Hate Speech Dataset — Overview

This directory describes the dataset used to train the SG-finetuned SingBERT model (`shareeep/singbert-hatexplain-and-syntheticsghate`). **The raw JSONL files are intentionally not part of the public repo** — they exist locally for the demo and the training notebooks in `notebooks/narrative_finals/`, but are gitignored.

If you need the data, ask the project owner.

---

## Schema

Each row is one JSON object with five string fields:

```json
{
  "text":             "Stop acting like a cheeko pek and leave those women alone.",
  "label":            "offensive",
  "target_group":     "none",
  "source_term":      "cheeko pek",
  "source_category":  "insults"
}
```

| Field | Values |
|---|---|
| `text` | One Singlish utterance, synthetically generated using Gemini |
| `label` | `hate` / `offensive` / `normal` |
| `target_group` | `chinese` / `malay` / `indian` / `none` |
| `source_term` | One of 20 Singlish terms (see list below) |
| `source_category` | `insults` (most rows) or blank (control / normal rows) |

---

## Splits

The full set was partitioned 60 / 20 / 20 for the training runs. The split JSONLs are not shipped — only the consolidated set is loaded by the demo locally (`data/singlish/synthetic/20_terms/consolidated.jsonl`, gitignored). Reproducing the splits is deterministic given the consolidated file + the seeded splitter in `archive/data/singlish/synthetic/wip-files/split_dataset_60_20_20.py`.

| Split | Rows | hate | offensive | normal |
|---|---:|---:|---:|---:|
| train | 319 | 108 | 99 | 112 |
| val   | 106 | 36 | 33 | 37 |
| test  | 108 | 37 | 33 | 38 |
| **total** | **533** | **181** | **165** | **187** |

Test split is held-out — every reported test metric in `results/model_scores.csv` is on these 108 rows. The demo's `sample_loader.py` samples from the consolidated set (it surfaces representative *demo moments*, not test-set rigour), so its picks may overlap with training rows; this is fine for the Arena page (free input) but means the Challenge page's player-vs-model verdicts are illustrative, not benchmarked.

### Target-group distribution (full set)

| target | count |
|---|---:|
| none | 349 |
| indian | 65 |
| chinese | 61 |
| malay | 58 |

Roughly two-thirds of rows have no specific target (`none`) — these are the offensive-but-not-hate and normal-control rows. The remaining third are racial hate examples balanced across the three main ethnic groups.

---

## The 20 source terms

Each of the 20 Singlish terms appears in every split. The terms span racial slurs, gendered insults, and culturally specific expletives. They were chosen to cover Singlish hate-speech patterns that English-only classifiers (HateXplain RoBERTa, generic toxicity models) systematically miss.

| Term | Rows | Notes |
|---|---:|---|
| ah kua | 36 | transphobic |
| anneh | 24 | racial — Indian male |
| apu neh neh | 17 | racial — Indian |
| big fuck | 28 | generic expletive |
| blur cock | 25 | sexual insult |
| bo lan pa | 22 | sexual insult |
| CB kia | 32 | strong expletive |
| cheeko pek | 34 | sexual harasser |
| cina babi | 23 | racial — Chinese |
| fatty bom bom | 23 | body-shaming |
| goondu | 27 | "fool" / dim-witted |
| jiak sai | 27 | scatological |
| keling | 18 | racial — Indian |
| lan jiao bin | 27 | sexual insult |
| na beh chee bai | 27 | strong expletive |
| pukimak | 31 | racial — Malay |
| rabak | 24 | "broken" — context dependent |
| siao cha bor | 26 | gendered insult — woman |
| siao lang | 26 | "crazy person" |
| Zhu Bajie | 36 | piggy-character body insult |

**Note on context-sensitivity:** terms like `ah kua`, `rabak`, `siao lang`, and `Zhu Bajie` appear across all three labels — the same term can be hate (targeting a group), offensive (general venting), or normal (descriptive / pedagogical). This is the false-positive trap the model has to navigate.

---

## Sample rows (held-out test set)

Two examples per label, picked for term diversity:

**`hate`** — racial target, slur deployed against the group
> *"Malay Singaporeans are all siao lang and should be locked up because they are a danger to society."* &nbsp;`target=malay  term=siao lang`
>
> *"Malay Singaporeans are a burden to this society and should just jiak sai instead of taking our resources."* &nbsp;`target=malay  term=jiak sai`

**`offensive`** — vulgar / insulting but not targeting a protected group
> *"That CB kia cut my lane without signaling then still want to stare at me, really no manners at all."* &nbsp;`target=none  term=CB kia`
>
> *"Stop acting like a cheeko pek and leave those women alone."* &nbsp;`target=none  term=cheeko pek`

**`normal`** — contains a known-bad term but in benign / referential context
> *"Uncle keep saying his neighbour body like Zhu Bajie, eat finish nothing to do, always disturbing people."* &nbsp;`target=none  term=Zhu Bajie`
>
> *"Apu neh neh is a slur that we should never use, really insensitive one."* &nbsp;`target=none  term=apu neh neh`

Notice how the same vocabulary appears across all three labels — context flips the verdict. This is what the SG-finetuned model learned to handle that generic toxicity classifiers cannot.

---

## How the data was generated

Synthetic dataset created with Gemini in a controlled pipeline:

1. **Term selection** — 20 high-signal Singlish terms picked from a scraped corpus (see `archive/notebooks/synthetic/`).
2. **Blueprint-driven generation** — for each term × (label × target_group) cell, Gemini was prompted to generate N rows following a blueprint.
3. **LLM-as-judge validation** — every row re-scored by a separate Gemini call; flagged rows were either repaired or removed.
4. **Manual policy review** — see `docs/synthetic_dataset_repair_decisions.md` for the term-by-term decisions made during cleanup.

The cleanup, regeneration, and judge-validation notebooks are archived at `archive/notebooks/synthetic/`.

---

## Why the data is not in the public repo

Singlish hate-speech examples include racial slurs targeting specific Singaporean communities. Even though the data is **synthetic**, broad distribution risks normalising or recirculating the language. The model checkpoint on HuggingFace (`shareeep/singbert-hatexplain-and-syntheticsghate`) is the artifact meant for sharing — the training data stays gated.

The `.gitignore` excludes `data/singlish/**/*.jsonl`. Cloning the repo gives you the model + the documentation, not the corpus.
