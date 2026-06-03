"""
Sample Browser — browse and filter the full held-out test split.
Accessible from the Streamlit sidebar nav when running demo/app.py.
"""
import json
import pathlib
import sys

import pandas as pd
import streamlit as st

# ── resolve demo/ modules ──────────────────────────────────────────────────────
_DEMO = pathlib.Path(__file__).parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from sample_loader import (
    BORDERLINE_TERMS,
    FALSE_POS_TERMS,
    SINGLISH_SPECIFIC,
    TEST_PATH,
)

st.title("Test Set — Sample Browser")
st.caption(f"Source: `{TEST_PATH.relative_to(_DEMO.parent)}`")

# ── load ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_df() -> pd.DataFrame:
    rows = [json.loads(l) for l in TEST_PATH.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)

    # tag demo bucket
    def bucket(r):
        if r["label"] == "hate" and r["source_term"] in SINGLISH_SPECIFIC:
            return "singlish_hate"
        if r["label"] == "hate" and r["target_group"] != "none":
            return "clear_hate"
        if r["label"] == "offensive" and r["source_term"] in BORDERLINE_TERMS:
            return "borderline"
        if r["label"] == "normal" and r["source_term"] in FALSE_POS_TERMS:
            return "false_pos"
        if r["label"] == "offensive" and r["target_group"] == "none":
            return "offensive_ok"
        return "—"

    df["demo_bucket"] = df.apply(bucket, axis=1)
    return df

df = load_df()

# ── filters ────────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    label_filter = st.multiselect(
        "Label",
        options=sorted(df["label"].unique()),
        default=sorted(df["label"].unique()),
    )
with col_b:
    bucket_filter = st.multiselect(
        "Demo bucket",
        options=sorted(df["demo_bucket"].unique()),
        default=sorted(df["demo_bucket"].unique()),
    )
with col_c:
    term_filter = st.multiselect(
        "Source term",
        options=sorted(df["source_term"].unique()),
        default=[],
        placeholder="All terms",
    )

search = st.text_input("Search text", placeholder="filter by keywords…")

# ── apply filters ──────────────────────────────────────────────────────────────
filtered = df[df["label"].isin(label_filter) & df["demo_bucket"].isin(bucket_filter)]
if term_filter:
    filtered = filtered[filtered["source_term"].isin(term_filter)]
if search:
    filtered = filtered[filtered["text"].str.contains(search, case=False, na=False)]

st.caption(f"{len(filtered)} / {len(df)} rows")

# ── colour-code label ──────────────────────────────────────────────────────────
LABEL_COLOURS = {"hate": "🔴", "offensive": "🟡", "normal": "🟢"}
BUCKET_LABELS = {
    "clear_hate":    "Clear hate",
    "singlish_hate": "Singlish hate",
    "offensive_ok":  "Offensive / ok",
    "false_pos":     "False pos trap",
    "borderline":    "Borderline",
    "—":             "—",
}

display = filtered.copy()
display["label"] = display["label"].map(lambda l: f"{LABEL_COLOURS.get(l, '')} {l}")
display["demo_bucket"] = display["demo_bucket"].map(BUCKET_LABELS)
display = display.rename(columns={
    "text": "Text",
    "label": "Label",
    "source_term": "Term",
    "target_group": "Target",
    "demo_bucket": "Demo bucket",
})

st.dataframe(
    display[["Text", "Label", "Term", "Target", "Demo bucket"]],
    use_container_width=True,
    hide_index=True,
)
