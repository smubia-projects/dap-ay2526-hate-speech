"""CPU-only Hugging Face inference service for the Singlish hate classifier."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_REPO_ID = "shareeep/singbert-hatexplain-and-syntheticsghate"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


class ModelService:
    """Load the classifier once and expose thread-safe, stateless predictions."""

    def __init__(self) -> None:
        configured_repo = os.getenv("MODEL_REPO_ID", "").strip()
        if not configured_repo:
            LOGGER.warning(
                "MODEL_REPO_ID is not set; falling back to public model %s",
                DEFAULT_MODEL_REPO_ID,
            )
        self.repo_id = configured_repo or DEFAULT_MODEL_REPO_ID
        self.device = torch.device("cpu")
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.hate_label_index: int | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _download_model(self) -> Path:
        paths = [
            Path(hf_hub_download(repo_id=self.repo_id, filename=filename))
            for filename in MODEL_FILES
        ]
        return paths[0].parent

    @staticmethod
    def _find_hate_label_index(config: Any) -> int:
        labels = {
            int(index): str(label).lower()
            for index, label in getattr(config, "id2label", {}).items()
        }
        for index, label in labels.items():
            if label == "hate" or "hate" in label:
                return index
        if getattr(config, "num_labels", 0) == 2:
            LOGGER.warning("No explicit hate label found; using binary label index 1")
            return 1
        raise RuntimeError("The model config does not identify a hate-speech label")

    def load(self) -> None:
        try:
            snapshot_dir = self._download_model()
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                snapshot_dir, local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.hate_label_index = self._find_hate_label_index(self.model.config)
            self.load_error = None
            LOGGER.info("Loaded %s on CPU", self.repo_id)
        except Exception as exc:
            self.tokenizer = None
            self.model = None
            self.hate_label_index = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Could not load model %s", self.repo_id)

    def predict(self, text: str, threshold: float) -> dict[str, Any]:
        if not self.loaded or self.hate_label_index is None:
            raise RuntimeError(self.load_error or "Model is not loaded")

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            hate_probability = float(probabilities[self.hate_label_index].item())

        review_floor = max(0.0, threshold - 0.2)
        verdict = (
            "block"
            if hate_probability >= threshold
            else "review"
            if hate_probability >= review_floor
            else "allow"
        )
        return {
            "hate_probability": round(hate_probability, 6),
            "verdict": verdict,
            "threshold": threshold,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_repo_id": self.repo_id,
        }
