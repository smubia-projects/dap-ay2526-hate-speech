"""
SinglishHateSpeech validator — extracted from notebooks/finals-phase2/guardrails/singlish_guardrails_demo.ipynb

Stripped of all Colab-specific code (google.colab, !pip, !guardrails configure,
@register_validator). Safe to import directly in any Python environment.
"""
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_model_cache: Dict[str, tuple] = {}


def load_classifier(model_name: str):
    """Load and cache a HuggingFace sequence classification model."""
    if model_name not in _model_cache:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
        print(f"  Labels: {model.config.id2label}")
    return _model_cache[model_name]


class SinglishHateSpeech:
    """
    Detects Singlish hate speech using a HuggingFace sequence classification model.

    Args:
        model_name: HuggingFace model ID. Defaults to SG_MODEL_ID env var or the
                    singbert fine-tuned model.
        hate_threshold: Confidence threshold for the hate label [0, 1].
        hate_label: Label name the model uses for hate speech (verify via
                    model.config.id2label before setting).
    """

    DEFAULT_MODEL = os.environ.get(
        "SG_MODEL_ID",
        "smubiainfrastructure/dap-ay2526-hate-speech-model",
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        hate_threshold: float = 0.6,
        hate_label: str = "hate",
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.hate_threshold = hate_threshold
        self.hate_label = hate_label
        self._tokenizer, self._model = load_classifier(self.model_name)

    def _get_hate_prob(self, text: str) -> float:
        """Run inference and return the probability for the hate label."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        label_to_id = {v: k for k, v in self._model.config.id2label.items()}
        hate_idx = label_to_id.get(self.hate_label, 0)
        return probs[hate_idx].item()

    def validate(self, text: str) -> dict:
        """
        Validate text for Singlish hate speech.

        Returns:
            {"passed": bool, "score": float, "fix_value": str | None}
        """
        if not isinstance(text, str) or not text.strip():
            return {"passed": True, "score": 0.0, "fix_value": None}

        score = self._get_hate_prob(text)
        passed = score < self.hate_threshold
        return {
            "passed": passed,
            "score": score,
            "fix_value": "[Content removed: hate speech detected]" if not passed else None,
        }
