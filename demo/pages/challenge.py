"""
Challenge — you're the moderator. 5 rounds, you-vs-the-guards-vs-ground-truth.
"""
import html
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ── resolve demo/ modules ──────────────────────────────────────────────────────
_DEMO = pathlib.Path(__file__).parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from guard import BACKEND_LABELS, KNOWN_BACKENDS, backend_status, run_guard
from sample_loader import get_demo_samples


def _run_backend(args: tuple[str, str]) -> tuple[str, dict]:
    backend, text = args
    ok, reason = backend_status(backend)
    if not ok:
        return backend, {"verdict": None, "skip_reason": reason}

    result = run_guard(text, backend=backend)
    if result.get("error"):
        return backend, {"verdict": None, "error": result["error"]}

    return backend, {
        "verdict": not result["passed"],   # True = block
        "score": result["score"],
        "fine_grained": result.get("fine_grained"),
        "verdict_label": result.get("verdict_label"),
        "latency_ms": result["latency_ms"],
    }

# ── header ─────────────────────────────────────────────────────────────────────
st.title("Challenge — You vs. the Guards")
st.caption(
    "5 rounds. For each message, decide: **Block**, **Allow**, or **Queue for review**. "
    "Then see the ground truth and how each guard ruled."
)

# ── session state ──────────────────────────────────────────────────────────────
def _new_game(seed: int | None = None):
    st.session_state["chal_samples"] = get_demo_samples(seed=seed)
    st.session_state["chal_round"] = 0
    st.session_state["chal_history"] = []
    st.session_state["chal_revealed"] = False
    st.session_state["chal_guard_results"] = {}

if "chal_samples" not in st.session_state:
    _new_game()

samples = st.session_state["chal_samples"]
total_rounds = len(samples)
idx = st.session_state["chal_round"]

# ── end of game ────────────────────────────────────────────────────────────────
if idx >= total_rounds:
    st.subheader("Game over")
    history = st.session_state["chal_history"]

    # tallies
    player_correct = sum(1 for h in history if h["player_correct"])
    backend_correct = {b: 0 for b in KNOWN_BACKENDS}
    backend_played = {b: 0 for b in KNOWN_BACKENDS}
    for h in history:
        for b, g in h["guards"].items():
            if g.get("verdict") is None:   # unavailable / error
                continue
            backend_played[b] += 1
            if g["verdict"] == h["ground_truth_block"]:
                backend_correct[b] += 1

    st.markdown(f"### You: **{player_correct} / {total_rounds}** correct")
    st.divider()
    st.markdown("**Guards:**")
    rows = []
    for b in KNOWN_BACKENDS:
        if backend_played[b] == 0:
            rows.append(f"- `{BACKEND_LABELS[b]:25}` — _not available this session_")
        else:
            rows.append(
                f"- `{BACKEND_LABELS[b]:25}` — **{backend_correct[b]} / {backend_played[b]}**"
            )
    st.markdown("\n".join(rows))

    st.divider()
    with st.expander("Round-by-round breakdown"):
        for i, h in enumerate(history, 1):
            st.markdown(
                f"**Round {i}** — gt: `{h['gt_label']}` "
                f"({'block' if h['ground_truth_block'] else 'allow'}) — "
                f"you: `{h['player_action']}` "
                f"({'✅' if h['player_correct'] else '❌'})"
            )
            st.caption(f"\"{h['text']}\"")

    if st.button("Play again", type="primary"):
        _new_game()
        st.rerun()
    st.stop()

# ── current round ──────────────────────────────────────────────────────────────
sample = samples[idx]
st.markdown(f"### Round {idx + 1} / {total_rounds}")
st.markdown(
    "<div style='background:#1d2230;color:#e6e9ef;padding:1rem 1.25rem;"
    "border-radius:8px;border-left:4px solid #5b8def;font-size:1.1rem;'>"
    f"{html.escape(sample['text'])}</div>",
    unsafe_allow_html=True,
)
st.caption("You're moderating a chat. What's your call?")

# ── choice / reveal flow ───────────────────────────────────────────────────────
revealed = st.session_state.get("chal_revealed", False)

if not revealed:
    c1, c2, c3 = st.columns(3)
    chosen = None
    with c1:
        if st.button("🛑 Block", width="stretch"):
            chosen = "block"
    with c2:
        if st.button("✅ Allow", width="stretch"):
            chosen = "allow"
    with c3:
        if st.button("🔍 Queue", width="stretch"):
            chosen = "queue"

    if chosen:
        # run every available guard on this sample, cache the verdicts
        guards = {}
        progress = st.progress(0.0, text="Asking the guards…")
        with ThreadPoolExecutor(max_workers=len(KNOWN_BACKENDS)) as pool:
            futures = [
                pool.submit(_run_backend, (backend, sample["text"]))
                for backend in KNOWN_BACKENDS
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                backend, result = future.result()
                guards[backend] = result
                progress.progress(
                    completed / len(KNOWN_BACKENDS),
                    text=f"Finished {completed} / {len(KNOWN_BACKENDS)} guards…",
                )
        progress.empty()

        ground_truth_block = sample["label"] == "hate"      # block ⇔ hate
        player_block = chosen == "block"
        # treat "queue" as halfway-correct only when ground truth is hate (defensible call)
        if chosen == "queue":
            player_correct = ground_truth_block            # cautious queue ≈ catch
        else:
            player_correct = player_block == ground_truth_block

        st.session_state["chal_guard_results"] = guards
        st.session_state["chal_revealed"] = True
        st.session_state["chal_history"].append({
            "text":               sample["text"],
            "gt_label":           sample["label"],
            "gt_target":          sample.get("target_group"),
            "gt_term":            sample.get("source_term"),
            "ground_truth_block": ground_truth_block,
            "player_action":      chosen,
            "player_correct":     player_correct,
            "guards":             guards,
        })
        st.rerun()

else:
    last = st.session_state["chal_history"][-1]
    guards = last["guards"]
    gt_block = last["ground_truth_block"]
    you = last["player_action"]
    correct = last["player_correct"]

    st.markdown("---")
    icon = "✅" if correct else "❌"
    expected = "block" if gt_block else "allow"
    st.markdown(
        f"**Your call:** `{you}`  &nbsp; **Ground truth:** `{last['gt_label']}` "
        f"(target: `{last['gt_target']}`, term: `{last['gt_term']}`) "
        f"→ should `{expected}` {icon}"
    )
    st.markdown("##### How each guard ruled")

    for b in KNOWN_BACKENDS:
        g = guards.get(b, {})
        label = BACKEND_LABELS[b]
        if g.get("verdict") is None:
            reason = g.get("skip_reason") or g.get("error") or "n/a"
            st.markdown(f"- `{label:25}` — _{reason}_")
            continue
        agree = g["verdict"] == gt_block
        agree_icon = "✅" if agree else "❌"
        verdict_str = "🛑 BLOCK" if g["verdict"] else "✅ ALLOW"
        score_str = (
            f"`{g['verdict_label']}`" if g.get("verdict_label")
            else f"score `{g['score']:.2f}`"
        )
        st.markdown(
            f"- `{label:25}` {verdict_str} · {score_str} · "
            f"{g['latency_ms']:.0f} ms · {agree_icon} vs ground truth"
        )

    st.divider()
    if st.button("Next round →", type="primary"):
        st.session_state["chal_round"] += 1
        st.session_state["chal_revealed"] = False
        st.session_state["chal_guard_results"] = {}
        st.rerun()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Challenge")
    st.markdown(f"Round **{min(idx + 1, total_rounds)}** of **{total_rounds}**")
    if st.button("Reset game"):
        _new_game()
        st.rerun()
    st.caption(
        "Samples are drawn from the 5 demo buckets in `sample_loader.py`: "
        "clear hate, Singlish hate, offensive-but-ok, false-positive trap, borderline."
    )
    st.caption(
        "**Caveat:** the samples may overlap with the SG model's training set, "
        "so its verdicts are illustrative, not benchmarked."
    )
