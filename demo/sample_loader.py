"""
sample_loader.py — filters the consolidated dataset into 5 demo-moment categories
and returns one random sample per category.
"""
import json
import pathlib
import random
from typing import Optional

# Relative to repo root — adjust if running from demo/ directly
_HERE = pathlib.Path(__file__).parent
DATA_PATH = _HERE.parent / "data/singlish/synthetic/20_terms/consolidated.jsonl"

SINGLISH_SPECIFIC = {"apu neh neh", "cina babi", "anneh", "pukimak"}
FALSE_POS_TERMS = {"ah kua", "rabak", "siao lang"}
BORDERLINE_TERMS = {"ah kua", "Zhu Bajie", "siao cha bor"}


def load_rows() -> list:
    return [json.loads(l) for l in DATA_PATH.read_text().splitlines() if l.strip()]


def get_demo_samples(seed: Optional[int] = None) -> list:
    """
    Return one sample from each of five demo-moment buckets, in order:
      1. clear_hate    — unambiguous racial hate, establishes credibility
      2. singlish_hate — Singlish-specific slurs the model must catch
      3. offensive_ok  — offensive but not hate, guard should pass
      4. false_pos     — normal text containing known-bad terms, tests context
      5. borderline    — scores in 0.4–0.65 range, motivates threshold discussion
    """
    rows = load_rows()
    rng = random.Random(seed)

    buckets = {
        "clear_hate": [
            r for r in rows
            if r["label"] == "hate"
            and r["target_group"] != "none"
            and r["source_term"] not in SINGLISH_SPECIFIC
        ],
        "singlish_hate": [
            r for r in rows
            if r["label"] == "hate" and r["source_term"] in SINGLISH_SPECIFIC
        ],
        "offensive_ok": [
            r for r in rows
            if r["label"] == "offensive" and r["target_group"] == "none"
        ],
        "false_pos": [
            r for r in rows
            if r["label"] == "normal" and r["source_term"] in FALSE_POS_TERMS
        ],
        "borderline": [
            r for r in rows
            if r["label"] == "offensive" and r["source_term"] in BORDERLINE_TERMS
        ],
    }

    selected = []
    used_texts = set()
    for bucket in buckets.values():
        if not bucket:
            continue
        row = rng.choice(bucket)
        selected.append(row)
        used_texts.add(row.get("text"))

    fallback_pool = [r for r in rows if r.get("text") not in used_texts]
    while len(selected) < len(buckets) and fallback_pool:
        idx = rng.randrange(len(fallback_pool))
        row = fallback_pool.pop(idx)
        selected.append(row)
        used_texts.add(row.get("text"))

    return selected
