"""
Leaderboard — model scores from results/model_scores.csv + your session vote tally.
"""
import pathlib
import sys

import pandas as pd
import streamlit as st

_DEMO = pathlib.Path(__file__).parent.parent
_ROOT = _DEMO.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from guard import BACKEND_LABELS, KNOWN_BACKENDS

SCORES_PATH = _ROOT / "results" / "model_scores.csv"

st.title("Leaderboard")
st.caption(
    "Held-out test set metrics from `results/model_scores.csv` "
    "plus your session vote tally from the Arena page."
)

# ── model scores ───────────────────────────────────────────────────────────────
st.subheader("Model scores")

if not SCORES_PATH.exists():
    st.warning(f"`{SCORES_PATH.relative_to(_ROOT)}` not found.")
else:
    df = pd.read_csv(SCORES_PATH)
    headline_cols = [
        "model_name", "model_type", "task",
        "accuracy", "macro_f1",
        "hatespeech_f1", "hatespeech_precision", "hatespeech_recall",
        "test_samples",
    ]
    available = [c for c in headline_cols if c in df.columns]
    view = df[available].copy()

    # nice column labels
    view = view.rename(columns={
        "model_name":           "Model",
        "model_type":           "Type",
        "task":                 "Task",
        "accuracy":             "Accuracy",
        "macro_f1":             "Macro F1",
        "hatespeech_f1":        "Hate F1",
        "hatespeech_precision": "Hate Prec.",
        "hatespeech_recall":    "Hate Recall",
        "test_samples":         "n",
    })

    # sort default by Hate F1 if present
    if "Hate F1" in view.columns:
        view = view.sort_values("Hate F1", ascending=False)

    st.dataframe(view, width="stretch", hide_index=True)

    with st.expander("All columns / raw"):
        st.dataframe(df, width="stretch", hide_index=True)

# ── session vote tally ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Your session — Arena votes")

votes: dict = st.session_state.get("arena_votes", {})
total_votes = sum(votes.values())

if total_votes == 0:
    st.info("No votes yet. Try the **Arena** page and tap “Lock in vote” after analysing a prompt.")
else:
    rows = []
    for b in KNOWN_BACKENDS:
        rows.append({
            "Backend": BACKEND_LABELS[b],
            "id":      b,
            "Votes":   votes.get(b, 0),
            "Share":   (votes.get(b, 0) / total_votes) if total_votes else 0.0,
        })
    vt = pd.DataFrame(rows).sort_values("Votes", ascending=False)
    st.dataframe(
        vt.style.format({"Share": "{:.0%}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Total votes this session: **{total_votes}**")

# ── session challenge tally ────────────────────────────────────────────────────
st.divider()
st.subheader("Your session — Challenge results")

history = st.session_state.get("chal_history", [])
if not history:
    st.info("No Challenge rounds played yet — head to the **Challenge** page.")
else:
    player_correct = sum(1 for h in history if h["player_correct"])
    total = len(history)
    st.markdown(f"**You:** {player_correct} / {total}")

    backend_correct = {b: 0 for b in KNOWN_BACKENDS}
    backend_played  = {b: 0 for b in KNOWN_BACKENDS}
    for h in history:
        for b, g in h["guards"].items():
            if g.get("verdict") is None:
                continue
            backend_played[b] += 1
            if g["verdict"] == h["ground_truth_block"]:
                backend_correct[b] += 1

    rows = []
    for b in KNOWN_BACKENDS:
        rows.append({
            "Backend":  BACKEND_LABELS[b],
            "Correct":  backend_correct[b],
            "Played":   backend_played[b],
            "Status":   "—" if backend_played[b] == 0 else f"{backend_correct[b]}/{backend_played[b]}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.caption(
    "Session tallies reset on page reload — Streamlit session state is browser-tab-local."
)
