"""CPU-only Hugging Face inference for the Beat the Guard comparison demo.

Two classifiers score every message so the frontend can put them side by side:

- the **SG-Tuned Guard** — this project's fine-tuned SingBERT (a `hate` probability);
- the **Standard Guard** — a generic English toxicity model, as a baseline foil
  (a `toxicity` probability).

Both load from HF Hub at startup and run stateless on CPU. The SG-Tuned guard is the
product; the Standard guard is best-effort, so the demo still runs (degraded) if the
foil fails to load.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_SG_REPO_ID = "shareeep/singbert-hatexplain-and-syntheticsghate"
DEFAULT_REGULAR_REPO_ID = "s-nlp/roberta_toxicity_classifier"
MAX_TOKENS = 256


def _positive_index(config: Any, candidates: tuple[str, ...]) -> int:
    """Index of the label to report as the flagged probability.

    Matches the first candidate substring against the model's ``id2label`` (e.g.
    ``hate`` for SingBERT, ``toxic`` for the toxicity foil). Falls back to index 1
    for an unlabelled binary head.
    """
    labels = {
        int(index): str(label).lower()
        for index, label in getattr(config, "id2label", {}).items()
    }
    for want in candidates:
        for index, label in labels.items():
            if want in label:
                return index
    if getattr(config, "num_labels", 0) == 2:
        LOGGER.warning("No label matched %s; using binary index 1", candidates)
        return 1
    raise RuntimeError(f"Model config exposes no label matching {candidates}")


def verdict_for(probability: float, threshold: float) -> str:
    """Policy band: block at/above threshold, review within 0.2 below it, else allow."""
    review_floor = max(0.0, threshold - 0.2)
    if probability >= threshold:
        return "block"
    if probability >= review_floor:
        return "review"
    return "allow"


class Classifier:
    """One HF sequence-classification model, loaded once and scored thread-safely."""

    def __init__(
        self,
        key: str,
        name: str,
        repo_id: str,
        metric: str,
        positive_labels: tuple[str, ...],
    ) -> None:
        self.key = key
        self.name = name
        self.repo_id = repo_id
        self.metric = metric
        self._positive_labels = positive_labels
        self.device = torch.device("cpu")
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.positive_index: int | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_id)
            self.model.to(self.device)
            self.model.eval()
            self.positive_index = _positive_index(self.model.config, self._positive_labels)
            self.load_error = None
            LOGGER.info("Loaded %s (%s) on CPU", self.repo_id, self.key)
        except Exception as exc:  # noqa: BLE001 — surface any load failure as degraded state
            self.tokenizer = None
            self.model = None
            self.positive_index = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Could not load %s (%s)", self.repo_id, self.key)

    def score(self, text: str) -> tuple[float, float]:
        """Return (flagged probability, latency in ms) for one message."""
        if not self.loaded or self.positive_index is None:
            raise RuntimeError(self.load_error or f"{self.key} is not loaded")

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            probability = float(probabilities[self.positive_index].item())
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return probability, latency_ms


class GuardService:
    """The two guards behind the demo. The SG-Tuned guard is required; the foil isn't."""

    def __init__(self) -> None:
        self.sg = Classifier(
            key="sg",
            name="SG-Tuned Guard",
            repo_id=os.getenv("MODEL_REPO_ID", "").strip() or DEFAULT_SG_REPO_ID,
            metric="hate",
            positive_labels=("hate",),
        )
        self.regular = Classifier(
            key="regular",
            name="Standard Guard",
            repo_id=os.getenv("REGULAR_MODEL_REPO_ID", "").strip() or DEFAULT_REGULAR_REPO_ID,
            metric="toxicity",
            positive_labels=("toxic", "hate", "offensive", "abusive"),
        )
        self.guards = (self.sg, self.regular)

    def load(self) -> None:
        for guard in self.guards:
            guard.load()

    @property
    def loaded(self) -> bool:
        # The SG-Tuned guard is the product; the demo can run without the foil.
        return self.sg.loaded

    def status(self) -> dict[str, bool]:
        return {guard.key: guard.loaded for guard in self.guards}

    def predict(self, text: str, threshold: float) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for guard in self.guards:
            if not guard.loaded:
                continue
            probability, latency_ms = guard.score(text)
            results.append(
                {
                    "key": guard.key,
                    "name": guard.name,
                    "metric": guard.metric,
                    "probability": round(probability, 6),
                    "verdict": verdict_for(probability, threshold),
                    "latency_ms": latency_ms,
                    "model_repo_id": guard.repo_id,
                }
            )
        if not results:
            raise RuntimeError(self.sg.load_error or "No guard is loaded")
        return {"text": text, "threshold": threshold, "guards": results}
