# Synthetic Dataset Repair & Regeneration — Decision Log

This document records the rationale behind every significant decision made during the cleanup and regeneration of the 20-term synthetic Singlish hate speech dataset. It is intended for stakeholders reviewing the dataset construction process.

---

## Background

The initial dataset was generated using Gemini Flash with 20 Singlish/SG terms, producing 540 rows (180 per label: hate / offensive / normal, 180 per split: train / val / test). After an LLM-as-a-judge validation pass, several quality issues were identified that required a structured repair and regeneration workflow.

---

## Stage 1: LLM-as-a-Judge Validation

### What we did
A blind Gemini Flash judge independently labelled every row without seeing the assigned label. It returned:
- `judge_label` (hate / offensive / normal)
- `judge_target_group`
- `judge_confidence` (high / medium / low)
- `singlish_realism` (high / medium / low)
- `reasoning`

### Key findings
| Issue | Count |
|---|---|
| Overall agreement | 462 / 540 (85.6%) — Cohen's κ = 0.783 |
| Exact duplicates | 54 pairs |
| Near-duplicates (>0.85 similarity) | 178 pairs |
| Low realism rows | 22 |
| Target group mismatches (hate rows) | 11 / 180 |
| High-confidence disagreements | 78 |

### Problem terms (judge agreement < 70%)
| Term | Agreement | Main issue |
|---|---|---|
| `apu neh neh` | 51.9% | Offensive rows were meta-linguistic descriptions ("The term X is a slur...") — judge correctly called them normal |
| `big fuck` | 66.7% | Normal rows were still vulgar — judge called them offensive |
| `bo lan pa` | 66.7% | Same as above |
| `cina babi` | 66.7% | Offensive rows judged as normal |
| `keling` | 66.7% | Same as `cina babi` |

---

## Stage 2: Cleanup Policy

### Decision: Remove exact duplicates and cross-split text leakage
**Rationale:** Exact duplicate text across splits constitutes data leakage. A model trained on train and evaluated on test with identical rows would have inflated metrics. These were removed unconditionally.

**Result:** 149 rows removed (110 exact duplicates + 106 cross-split, deduplicated to 149 unique rows).

### Decision: Enable removal of high-confidence judge disagreements
**Rationale:** The judge flagged rows where it was confident the assigned label was wrong. The most systematic issue was `apu neh neh` offensive rows that were clearly meta-linguistic definitions (e.g. *"The term apu neh neh is a derogatory slur used in Singapore"*) — these are not examples of offensive use, they describe the term. Keeping them would introduce noise into model training.

We initially left this off but enabled it after the judge results confirmed the pattern was systematic, not random.

### Decision: Keep low-realism rows (not auto-deleted)
**Rationale:** Low realism (4.1% of rows) was used as a flag for human review rather than automatic deletion. Some low-realism rows may still be valid examples; the call was left to manual review via the dataset viewer.

---

## Stage 3: Term Policy

Each of the 20 terms was assigned a generation policy based on its definitions and usage notes:

| Policy | Meaning | Terms |
|---|---|---|
| `hate_only_candidate` | Term is a racial slur — only hate-labelled examples are realistic | `apu neh neh`, `keling` |
| `hate_dominant` | Term is heavily loaded but limited non-hate use is possible | `cina babi`, `jiak sai` |
| `offensive_dominant` | Term is vulgar/insulting but does not target a protected group | `blur cock`, `bo lan pa`, `CB kia`, etc. |
| `flexible` | Term can appear naturally across all three labels | `goondu`, `rabak`, `siao lang`, etc. |

### Decision: Add `identity_linked` flag for racially-motivated terms
**Rationale:** Four terms are intrinsically tied to a specific racial/ethnic identity — `anneh` (Indian men), `apu neh neh` (Indians), `cina babi` (Chinese), `keling` (Indians/Tamils). Even when used in an "offensive" or "normal" sentence, the term itself invokes the target group, making the label ambiguous. For example, `anneh` used as a generic insult still targets Indian men by definition.

**Effect:** These terms are excluded from non-hate gap-filling entirely. No new offensive or normal rows are generated for them. Any non-hate rows already in the cleaned dataset that passed the judge check are retained.

**Why not delete all their non-hate rows?** Only high-confidence disagreements were removed. Some non-hate rows for these terms (e.g. clearly academic/quoted uses) were agreed upon by the judge and remain valid.

---

## Stage 4: Regeneration Strategy

### Hate rows: Blueprint-driven regeneration
**Rationale:** Hate rows were regenerated using the exact deleted rows as blueprints — preserving the same `(term, target_group, split)` distribution that was lost. This ensures the repaired dataset maintains the same target group balance as originally intended.

**Cross-group concern with `cina babi`:** The term literally means "Chinese pig" but some original rows had `target_group=malay` or `target_group=indian`. These were flagged for manual review and corrected before regen where applicable.

### Non-hate rows: Gap-driven with term cycling
**Rationale:** Offensive and normal gaps were filled by cycling across the remaining eligible terms (excluding `hate_only_candidate` and identity-linked terms). This distributes the regeneration load across general-purpose Singlish insults rather than forcing generation for terms where non-hate use is linguistically unrealistic.

### Prompt styles
| Style | When used |
|---|---|
| `term_conditioned_hate` | All hate rows — term must appear in text, target group must be mentioned |
| `term_conditioned_non_hate` | Offensive/normal rows for `flexible` and `offensive_dominant` terms |
| `generic_singlish_non_hate` | Offensive rows for `hate_dominant` / identity-linked terms |
| `generic_singlish_normal` | Normal rows for `hate_dominant` / identity-linked terms |

### Validation on each generated row
Every generated row was validated before being written:
- Label must match the requested label
- For term-conditioned prompts: term must appear in the text
- For hate rows: assigned `target_group` must match, and the target group must be explicitly mentioned in the text
- Implied group keywords were added for identity-linked terms (e.g. `keling` in text counts as mentioning the `indian` group)

### Deduplication at merge
Before merging regenerated rows into the final splits, a deduplication check was run against the cleaned base rows and within the regen batch itself. Duplicate rows were removed and retried.

---

## Stage 5: Manual Review

A Streamlit dataset viewer (`scripts/view_dataset.py`) was built to allow manual inspection and correction of flagged rows. Any manually edited row is tagged with:
- `manually_edited: true`
- `edit_rationale`: the reviewer's written justification

This provides an audit trail for any post-hoc label corrections.

---

## Final Dataset Stats

| Split | Rows |
|---|---|
| train | 180 |
| val | 180 |
| test | 180 |
| **Total** | **540** |

Label distribution: 180 hate / 180 offensive / 180 normal (perfectly balanced).

---

## Files

| File | Purpose |
|---|---|
| `repair_outputs/delete_candidates.json` | All rows flagged for deletion and why |
| `repair_outputs/term_policy.json` | Per-term generation policy and identity_linked flag |
| `repair_outputs/removed_hate_blueprints.json` | Deleted hate rows used as regen blueprints |
| `repair_outputs/post_cleanup_summary.json` | Label/split gap counts after cleanup |
| `repair_outputs/cleaned_splits/` | Post-cleanup base splits before regeneration |
| `repair_outputs/regenerated_outputs/regenerated_rows.jsonl` | All successfully generated rows |
| `repair_outputs/regenerated_outputs/repaired_splits/` | Final repaired dataset |
