"""
Arena — type any input, see how every guard backend scores it side-by-side.
"""
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ── resolve demo/ modules ──────────────────────────────────────────────────────
_DEMO = pathlib.Path(__file__).parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from guard import BACKEND_LABELS, KNOWN_BACKENDS, backend_status, run_guard


def _run_backend(args: tuple[str, str, float]) -> dict:
    backend, text, threshold = args
    ok, reason = backend_status(backend)
    if not ok:
        return {
            "backend": backend,
            "available": False,
            "skip_reason": reason,
        }

    result = run_guard(text, threshold=threshold, backend=backend)
    result["available"] = True
    return result

# ── header ─────────────────────────────────────────────────────────────────────
st.title("Arena — Side-by-side guard comparison")
st.caption(
    "Type a Singlish prompt. Every available guard backend scores it. "
    "Backends without API keys show as unavailable — set the keys in `demo/.env`."
)

# ── session state ──────────────────────────────────────────────────────────────
if "arena_input" not in st.session_state:
    st.session_state["arena_input"] = ""
if "arena_results" not in st.session_state:
    st.session_state["arena_results"] = []  # list[dict] per backend
if "arena_votes" not in st.session_state:
    # tally: {backend_id: int}
    st.session_state["arena_votes"] = {b: 0 for b in KNOWN_BACKENDS}

# ── controls ───────────────────────────────────────────────────────────────────
PRESETS = {
    "— pick a preset —": "",
    "Singlish slur — direct hate":  "knn those cina babi think they own the place",
    "Borderline frustration":       "those ah neh always cut queue, sibei sian lah",
    "False-positive trap (benign)": "eh wanna makan at that ah kua stall later?",
    "Plain English hate":           "all asians should go back to their country",
    "Normal Singlish":              "wah lao, this kopi damn shiok one sia",
}

col_in, col_btns = st.columns([4, 1])
with col_in:
    preset_key = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")
    if PRESETS[preset_key] and st.session_state.get("arena_last_preset") != preset_key:
        st.session_state["arena_input"] = PRESETS[preset_key]
        st.session_state["arena_last_preset"] = preset_key

    text = st.text_area(
        "Input",
        key="arena_input",
        height=100,
        placeholder="Type or pick a preset…",
    )
    threshold = st.slider("Hate threshold (for SG / LionGuard)", 0.3, 0.9, 0.6, 0.05)

with col_btns:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze All", type="primary", width="stretch")
    if st.button("Clear", width="stretch"):
        st.session_state["arena_results"] = []
        st.rerun()

# ── run all backends ───────────────────────────────────────────────────────────
if analyze_clicked and text.strip():
    results_by_backend = {}
    progress = st.progress(0.0, text="Running backends…")
    with ThreadPoolExecutor(max_workers=len(KNOWN_BACKENDS)) as pool:
        futures = [
            pool.submit(_run_backend, (backend, text, threshold))
            for backend in KNOWN_BACKENDS
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results_by_backend[result["backend"]] = result
            progress.progress(
                completed / len(KNOWN_BACKENDS),
                text=f"Finished {completed} / {len(KNOWN_BACKENDS)} backends…",
            )
    progress.empty()
    st.session_state["arena_results"] = [
        results_by_backend[backend] for backend in KNOWN_BACKENDS
    ]

# ── render scoreboard ──────────────────────────────────────────────────────────
results = st.session_state["arena_results"]
if not results:
    st.info(
        "Enter a prompt and click **Analyze All**. The Arena calls every guard "
        "backend that has its API key configured."
    )
else:
    st.subheader("Verdicts")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        backend = r["backend"]
        label = BACKEND_LABELS[backend]

        with col:
            st.markdown(f"##### {label}")
            st.caption(f"`{backend}`")

            if not r.get("available"):
                st.warning(f"Unavailable — {r['skip_reason']}")
                continue

            if r.get("error"):
                st.error(f"Error: {r['error']}")
                continue

            # OVERALL verdict
            score = r["score"]
            passed = r["passed"]
            verdict_label = r.get("verdict_label")

            if verdict_label:                       # binary backend (SEA-Guard)
                if passed:
                    st.success(f"### ✅ {verdict_label}")
                else:
                    st.error(f"### 🛑 {verdict_label}")
            else:                                   # continuous backends
                if score >= 0.6:
                    st.error(f"### 🛑 Block  ·  {score:.2f}")
                elif score >= 0.4:
                    st.warning(f"### 🟡 Uncertain  ·  {score:.2f}")
                else:
                    st.success(f"### ✅ Allow  ·  {score:.2f}")
                st.progress(min(score, 1.0))

            # latency badge
            st.caption(f"⏱ {r['latency_ms']:.0f} ms")

            # fine-grained categories
            fine = r.get("fine_grained")
            if fine:
                with st.expander("Per-category breakdown"):
                    # canonical order — LionGuard's 11 categories
                    order = [
                        "binary",
                        "hateful_l1", "hateful_l2",
                        "insults",
                        "sexual_l1", "sexual_l2",
                        "physical_violence",
                        "self_harm_l1", "self_harm_l2",
                        "all_other_misconduct_l1", "all_other_misconduct_l2",
                    ]
                    for k in order:
                        if k not in fine:
                            continue
                        v = fine[k]
                        bar_colour = "🔴" if v >= 0.6 else ("🟡" if v >= 0.4 else "🟢")
                        st.markdown(f"{bar_colour} `{k:25}` **{v:.2f}**")

    # ── session vote / sanity-check widget ────────────────────────────────────
    available_now = [r for r in results if r.get("available") and not r.get("error")]
    if len(available_now) >= 2:
        st.divider()
        st.subheader("Which verdict do you trust most?")
        st.caption("Logged to your session tally — visible on the Leaderboard page.")
        choices = ["— skip —"] + [BACKEND_LABELS[r["backend"]] for r in available_now]
        pick = st.radio("Pick one", choices, horizontal=True, label_visibility="collapsed")
        if pick != "— skip —" and st.button("Lock in vote"):
            backend_id = next(
                r["backend"] for r in available_now if BACKEND_LABELS[r["backend"]] == pick
            )
            st.session_state["arena_votes"][backend_id] = (
                st.session_state["arena_votes"].get(backend_id, 0) + 1
            )
            st.success(f"Vote recorded for **{pick}**.")

# ── footer / framing ───────────────────────────────────────────────────────────
with st.expander("About the backends"):
    st.markdown(
        "- **SG SingBERT** — our SG-finetuned encoder, end-to-end (~110M params), "
        "binary hate/normal head. Runs locally.\n"
        "- **LLM Guard (Toxicity)** — generic toxicity scanner; not Singlish-aware. Runs locally.\n"
        "- **LionGuard 2.0** — GovTech, OpenAI embeddings + 849K-param classifier head. "
        "Needs `OPENAI_API_KEY`.\n"
        "- **LionGuard 2.1** — same head, Gemini embeddings. Needs `GEMINI_KEY`.\n"
        "- **SEA-Guard** — AI Singapore LLM-based judge via `api.sea-lion.ai`. "
        "Binary SAFE/UNSAFE only. Needs `SEA_LION_API_KEY`."
    )
