"""
guard.py — normalised guard interface.

Active guard model is selected via GUARD_MODEL env var at launch:
  GUARD_MODEL=sg          (default) SG fine-tuned SingBERT
  GUARD_MODEL=llmguard    LLM Guard Toxicity scanner
  GUARD_MODEL=lionguard   GovTech LionGuard v2 (requires OPENAI_API_KEY)

Returns a common dict: {"score": float, "passed": bool}
"""
import os

import streamlit as st

GUARD_MODEL = os.environ.get("GUARD_MODEL", "sg")


@st.cache_resource
def load_guard():
    if GUARD_MODEL == "sg":
        from singlish_validator import SinglishHateSpeech

        return SinglishHateSpeech(
            model_name=os.environ.get(
                "SG_MODEL_ID",
                "shareeep/singbert-hatexplain-and-syntheticsghate",
            ),
            hate_label=os.environ.get("HATE_LABEL", "hate"),
        )
    elif GUARD_MODEL == "llmguard":
        from llm_guard.input_scanners import Toxicity

        return Toxicity()
    elif GUARD_MODEL == "lionguard":
        from transformers import AutoModel

        return AutoModel.from_pretrained("govtech/lionguard-2", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown GUARD_MODEL: {GUARD_MODEL!r}. Use sg | llmguard | lionguard")


def run_guard(text: str, threshold: float = None) -> dict:
    """
    Run the active guard on text.

    Args:
        text:      Input text to classify.
        threshold: Override the default threshold (only applies to sg model).

    Returns:
        {"score": float, "passed": bool}
    """
    if threshold is None:
        threshold = float(os.environ.get("HATE_THRESHOLD", "0.6"))

    guard = load_guard()

    if GUARD_MODEL == "sg":
        score = guard._get_hate_prob(text)
        return {"score": score, "passed": score < threshold}

    elif GUARD_MODEL == "llmguard":
        _, is_valid, risk_score = guard.scan(prompt="", output=text)
        return {"score": float(risk_score), "passed": bool(is_valid)}

    elif GUARD_MODEL == "lionguard":
        from openai import OpenAI

        client = OpenAI()
        emb = (
            client.embeddings.create(input=text, model="text-embedding-3-large")
            .data[0]
            .embedding
        )
        result = guard.predict([emb])
        score = float(result["hateful_l2"][0])
        return {"score": score, "passed": score < threshold}
