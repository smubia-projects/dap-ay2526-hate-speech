"""
guard.py — normalised, swappable guard interface.

Backends:
  sg              SG fine-tuned SingBERT (local HF model)
  llmguard        LLM Guard Toxicity scanner (local)
  lionguard       GovTech LionGuard 2.0 (HF model + OpenAI embeddings) — needs OPENAI_API_KEY
  lionguard-2.1   GovTech LionGuard 2.1 (HF model + Gemini embeddings) — needs GEMINI_KEY
  sea-guard       AI Singapore SEA-Guard via api.sea-lion.ai — needs SEA_LION_API_KEY

GUARD_MODEL env var selects the default backend used by single-model pages
(chat.py). Multi-backend pages (arena.py, challenge.py) call run_guard()
with an explicit `backend=` argument.

run_guard() returns:
    {
        "score":         float,         # [0, 1] — primary "hate-ness" probability
        "passed":        bool,          # True = safe per `threshold`
        "fine_grained":  dict | None,   # per-category dict (LionGuard) or None
        "verdict_label": str | None,    # binary string verdict (SEA-Guard) or None
        "latency_ms":    float,         # measured backend latency
        "backend":       str,           # backend id that produced this
        "error":         str | None,    # human-readable error if call failed
    }
"""
import os
import time
import importlib.util
from typing import Optional

import streamlit as st

DEFAULT_GUARD = os.environ.get("GUARD_MODEL", "sg")
GUARD_MODEL = DEFAULT_GUARD  # legacy alias — existing imports still work
API_TIMEOUT_SECONDS = float(os.environ.get("GUARD_API_TIMEOUT_SECONDS", "20"))

KNOWN_BACKENDS = ("sg", "llmguard", "lionguard", "lionguard-2.1", "sea-guard")

BACKEND_LABELS = {
    "sg":            "SG SingBERT",
    "llmguard":      "LLM Guard (Toxicity)",
    "lionguard":     "LionGuard 2.0",
    "lionguard-2.1": "LionGuard 2.1",
    "sea-guard":     "SEA-Guard",
}


def _require_env(name: str, reason: str) -> tuple[bool, str]:
    return (True, "") if os.environ.get(name) else (False, reason)


def backend_status(backend: str) -> tuple[bool, str]:
    """
    Cheap pre-flight check — returns (available, reason_if_not).
    Does not load models; only checks required env keys.
    """
    if backend == "sg":
        return True, ""
    if backend == "llmguard":
        if importlib.util.find_spec("llm_guard") is None:
            return False, "llm-guard package is not installed"
        return True, ""
    if backend == "lionguard":
        return _require_env("OPENAI_API_KEY",
                            "needs OPENAI_API_KEY for text-embedding-3-large")
    if backend == "lionguard-2.1":
        return (True, "") if _gemini_key() else (
            False,
            "needs GEMINI_KEY or GEMINI_API_KEY for gemini-embedding-001",
        )
    if backend == "sea-guard":
        return _require_env("SEA_LION_API_KEY", "needs SEA_LION_API_KEY")
    return False, f"unknown backend {backend!r}"


@st.cache_resource(show_spinner=False)
def load_guard(backend: str):
    if backend == "sg":
        from singlish_validator import SinglishHateSpeech

        return SinglishHateSpeech(
            model_name=os.environ.get(
                "SG_MODEL_ID",
                "smubiainfrastructure/dap-ay2526-hate-speech-model",
            ),
            hate_label=os.environ.get("HATE_LABEL", "hate"),
        )

    if backend == "llmguard":
        from llm_guard.input_scanners import Toxicity
        return Toxicity()

    if backend == "lionguard":
        from transformers import AutoModel
        _patch_lionguard_transformers_compat()
        return AutoModel.from_pretrained("govtech/lionguard-2", trust_remote_code=True)

    if backend == "lionguard-2.1":
        from transformers import AutoModel
        _patch_lionguard_transformers_compat()
        return AutoModel.from_pretrained("govtech/lionguard-2.1", trust_remote_code=True)

    if backend == "sea-guard":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ["SEA_LION_API_KEY"],
            base_url="https://api.sea-lion.ai/v1",
            timeout=API_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unknown backend: {backend!r}. Use one of {KNOWN_BACKENDS}")


def _empty_result(backend: str, error: str, latency_ms: float = 0.0) -> dict:
    return {
        "score":         0.0,
        "passed":        True,
        "fine_grained":  None,
        "verdict_label": None,
        "latency_ms":    latency_ms,
        "backend":       backend,
        "error":         error,
    }


def _gemini_key() -> Optional[str]:
    return os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")


def _patch_lionguard_transformers_compat() -> None:
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def run_guard(
    text: str,
    threshold: Optional[float] = None,
    backend: Optional[str] = None,
) -> dict:
    """Run a guard backend on text. See module docstring for return shape."""
    if threshold is None:
        threshold = float(os.environ.get("HATE_THRESHOLD", "0.6"))
    backend = backend or DEFAULT_GUARD

    ok, reason = backend_status(backend)
    if not ok:
        return _empty_result(backend, reason)

    start = time.perf_counter()

    try:
        if backend == "sg":
            guard = load_guard("sg")
            score = guard._get_hate_prob(text)
            result = {
                "score":         score,
                "passed":        score < threshold,
                "fine_grained":  None,
                "verdict_label": None,
            }

        elif backend == "llmguard":
            guard = load_guard("llmguard")
            _, is_valid, risk_score = guard.scan(prompt="", output=text)
            result = {
                "score":         float(risk_score),
                "passed":        bool(is_valid),
                "fine_grained":  None,
                "verdict_label": None,
            }

        elif backend == "lionguard":
            from openai import OpenAI
            guard = load_guard("lionguard")
            client = OpenAI(timeout=API_TIMEOUT_SECONDS)
            emb = (
                client.embeddings.create(input=text, model="text-embedding-3-large")
                .data[0]
                .embedding
            )
            raw = guard.predict([emb])
            fine = {k: float(v[0]) for k, v in raw.items()}
            score = max(fine.get("hateful_l1", 0.0), fine.get("hateful_l2", 0.0))
            result = {
                "score":         score,
                "passed":        score < threshold,
                "fine_grained":  fine,
                "verdict_label": None,
            }

        elif backend == "lionguard-2.1":
            import numpy as np
            from google import genai

            guard = load_guard("lionguard-2.1")
            client = genai.Client(api_key=_gemini_key())
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[text],
            )
            embeddings = np.array([emb.values for emb in response.embeddings])
            raw = guard.predict(embeddings)
            fine = {k: float(v[0]) for k, v in raw.items()}
            score = max(fine.get("hateful_l1", 0.0), fine.get("hateful_l2", 0.0))
            result = {
                "score":         score,
                "passed":        score < threshold,
                "fine_grained":  fine,
                "verdict_label": None,
            }

        elif backend == "sea-guard":
            client = load_guard("sea-guard")
            response = client.chat.completions.create(
                model=os.environ.get("SEA_GUARD_MODEL", "aisingapore/SEA-Guard"),
                messages=[{"role": "user", "content": text}],
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
            is_unsafe = "UNSAFE" in verdict
            result = {
                "score":         1.0 if is_unsafe else 0.0,
                "passed":        not is_unsafe,
                "fine_grained":  None,
                "verdict_label": verdict or "UNKNOWN",
            }

        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return _empty_result(backend, f"{type(exc).__name__}: {exc}", latency_ms)

    result["latency_ms"] = (time.perf_counter() - start) * 1000
    result["backend"] = backend
    result["error"] = None
    return result
