import json
from pathlib import Path

import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIRED_DIR = (
    REPO_ROOT
    / "data/singlish/synthetic/20-terms"
    / "repair_outputs/regenerated_outputs/repaired_splits"
)
ORIGINAL_DIR = REPO_ROOT / "data/singlish/synthetic/20-terms"


st.set_page_config(page_title="Synthetic Dataset Viewer", layout="wide")
st.title("Synthetic Dataset Viewer")
st.caption("🔴 Hate &nbsp;|&nbsp; 🟠 Offensive &nbsp;|&nbsp; 🟢 Normal &nbsp;|&nbsp; ⚠️ Flagged by judge &nbsp;|&nbsp; ✏️ Manually edited")


@st.cache_data
def load_data():
    data_dir = REPAIRED_DIR if REPAIRED_DIR.exists() else ORIGINAL_DIR
    rows = []
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["split"] = split
                    row["row_index"] = idx
                    rows.append(row)
    return pd.DataFrame(rows), data_dir


@st.cache_data
def load_judge_results(data_dir):
    judge_file = data_dir / "judge_results.jsonl"
    if not judge_file.exists():
        judge_file = ORIGINAL_DIR / "judge_results.jsonl"
    if not judge_file.exists():
        return pd.DataFrame()
    rows = []
    with open(judge_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def save_edit(split, row_index, new_text, new_label, new_target_group, rationale):
    data_dir = REPAIRED_DIR if REPAIRED_DIR.exists() else ORIGINAL_DIR
    path = data_dir / f"{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if idx == row_index:
                row["text"] = new_text
                row["label"] = new_label
                row["target_group"] = new_target_group
                row["manually_edited"] = True
                row["edit_rationale"] = rationale
            rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    st.cache_data.clear()


df, data_dir = load_data()
judge_df = load_judge_results(data_dir)

has_judge = not judge_df.empty
if has_judge:
    judge_cols = judge_df[["text", "judge_label", "judge_target_group",
                            "judge_confidence", "singlish_realism", "agreement", "reasoning"]]
    df = df.merge(judge_cols, on="text", how="left")
    df["agreement"] = df["agreement"].fillna(True)
    df["flagged"] = (df["agreement"] == False) | (df["singlish_realism"] == "low")
else:
    df["flagged"] = False

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

splits = st.sidebar.multiselect("Split", ["train", "val", "test"], default=["train", "val", "test"])
labels = st.sidebar.multiselect("Label", ["hate", "offensive", "normal"], default=["hate", "offensive", "normal"])
terms = st.sidebar.multiselect("Source term", sorted(df["source_term"].unique()), default=[])
groups = st.sidebar.multiselect("Target group", sorted(df["target_group"].unique()), default=[])
search = st.sidebar.text_input("Search text")

if has_judge:
    show_flagged_only = st.sidebar.checkbox("Show flagged only", value=False)
else:
    show_flagged_only = False
    st.sidebar.info("No judge results found — run the judge notebook first to see flags.")

show_edited_only = st.sidebar.checkbox("Show manually edited only", value=False)

filtered = df[df["split"].isin(splits) & df["label"].isin(labels)]
if terms:
    filtered = filtered[filtered["source_term"].isin(terms)]
if groups:
    filtered = filtered[filtered["target_group"].isin(groups)]
if search:
    filtered = filtered[filtered["text"].str.contains(search, case=False, na=False)]
if show_flagged_only:
    filtered = filtered[filtered["flagged"] == True]
if show_edited_only:
    filtered = filtered[filtered.get("manually_edited", pd.Series(False, index=filtered.index)).fillna(False) == True]

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
filtered = filtered.copy()
filtered["_split_order"] = filtered["split"].map(SPLIT_ORDER)
filtered = filtered.sort_values(["source_term", "_split_order", "row_index"]).drop(columns="_split_order")

# ── Stats ──────────────────────────────────────────────────────────────────────
st.subheader(f"Showing {len(filtered):,} / {len(df):,} rows")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**By label**")
    st.dataframe(filtered.groupby("label").size().rename("count").reset_index(),
                 hide_index=True, use_container_width=True)
with col2:
    st.markdown("**By split**")
    st.dataframe(filtered.groupby("split").size().rename("count").reset_index(),
                 hide_index=True, use_container_width=True)
with col3:
    st.markdown("**By label × split**")
    st.dataframe(filtered.groupby(["split", "label"]).size().rename("count").reset_index(),
                 hide_index=True, use_container_width=True)

if has_judge:
    flagged_count = int(filtered["flagged"].sum())
    if flagged_count:
        n_disagree = int((filtered["agreement"] == False).sum())
        n_low_realism = int((filtered["singlish_realism"] == "low").sum())
        n_both = int(((filtered["agreement"] == False) & (filtered["singlish_realism"] == "low")).sum())
        parts = []
        if n_disagree:
            parts.append(f"**{n_disagree}** judge disagreement{'s' if n_disagree > 1 else ''}")
        if n_low_realism:
            parts.append(f"**{n_low_realism}** low realism")
        if n_both:
            parts.append(f"**{n_both}** with both issues")
        st.warning(f"⚠ {flagged_count} flagged rows in current view — {' · '.join(parts)}")

# ── Rows ───────────────────────────────────────────────────────────────────────
st.subheader("Rows")

LABEL_COLOURS = {"hate": "🔴", "offensive": "🟠", "normal": "🟢"}
ALL_LABELS = ["hate", "offensive", "normal"]
ALL_GROUPS = sorted(df["target_group"].unique())

for i, (_, row) in enumerate(filtered.iterrows()):
    icon = LABEL_COLOURS.get(row["label"], "⚪")
    flag = "⚠️ " if row.get("flagged") else ""
    edited = "✏️ " if row.get("manually_edited") else ""
    preview = row["text"][:80] + ("..." if len(row["text"]) > 80 else "")
    title = f"{flag}{edited}{icon} [{row['split']}] {row['source_term']} — {preview}"

    with st.expander(title, expanded=bool(row.get("flagged"))):

        # Manually edited badge
        if row.get("manually_edited"):
            rationale_text = row.get("edit_rationale", "").strip()
            if rationale_text:
                st.info(f"✏️ Manually edited — *{rationale_text}*")
            else:
                st.info("✏️ Manually edited")

        # Flag reasons
        if has_judge and row.get("flagged"):
            reasons = []
            if row.get("agreement") == False:
                reasons.append(f"❌ Label disagreement — judge says `{row.get('judge_label')}` (confidence: `{row.get('judge_confidence')}`)")
            if row.get("singlish_realism") == "low":
                reasons.append("🔴 Low Singlish realism")
            for r in reasons:
                st.markdown(r)
            if row.get("reasoning"):
                st.caption(f"Judge reasoning: {row['reasoning']}")
            st.divider()

        # Judge info for non-flagged rows
        if has_judge and pd.notna(row.get("judge_label")) and not row.get("flagged"):
            realism_icon = {"high": "✅", "medium": "🟡", "low": "🔴"}.get(row.get("singlish_realism"), "")
            st.caption(
                f"Judge: ✅ agree · confidence: `{row.get('judge_confidence')}` · "
                f"realism: {realism_icon} `{row.get('singlish_realism')}`"
            )

        # Current values
        edit_key = f"edit_{row['split']}_{row['row_index']}"
        st.markdown(f"**Term:** `{row['source_term']}`")

        new_text = st.text_area("Text", value=row["text"], key=f"text_{edit_key}", height=80)
        c1, c2 = st.columns(2)
        with c1:
            new_label = st.selectbox(
                "Label", ALL_LABELS,
                index=ALL_LABELS.index(row["label"]) if row["label"] in ALL_LABELS else 0,
                key=f"label_{edit_key}"
            )
        with c2:
            new_target_group = st.selectbox(
                "Target group", ALL_GROUPS,
                index=ALL_GROUPS.index(row["target_group"]) if row["target_group"] in ALL_GROUPS else 0,
                key=f"tg_{edit_key}"
            )

        existing_rationale = row.get("edit_rationale") or ""
        new_rationale = st.text_area(
            "Your edit rationale",
            value=existing_rationale,
            placeholder="Why are you making this change?",
            key=f"rationale_{edit_key}",
            height=60,
        )

        if st.button("💾 Save", key=f"save_{edit_key}"):
            save_edit(row["split"], row["row_index"], new_text, new_label, new_target_group, new_rationale)
            st.success("Saved — reloading...")
            st.rerun()
