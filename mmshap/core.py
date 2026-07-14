"""The MM-SHAP method, independent of any specific model.

MM-SHAP (Parcalabescu & Frank, ACL 2023) quantifies how much a vision-and-language
model relies on the text vs. the image when scoring an (image, sentence) pair:

  1. treat each text token and each image patch as a feature,
  2. use SHAP to attribute the model's image-text matching score to each feature,
  3. MM-score = Σ|text attributions| / Σ|all attributions|  (the textual share; the
     visual share is 1 - MM-score).

A concrete model plugs in by implementing :class:`MMShapModel`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shap
import torch

from .masking import mask_features, patch_grid_size, zero_patches


@dataclass
class Sample:
    """A preprocessed (image, sentence) pair ready to be probed by MM-SHAP."""
    input_ids: torch.Tensor    # (1, n_text_ids) text token ids fed to the model
    n_text: int                # text feature count (grid sizing + text/image split)
    image: torch.Tensor        # (1, 3, H, W) image tensor whose patches get masked
    patch_h: int               # patch height in pixels
    patch_w: int               # patch width in pixels
    grid_cols: int             # patches per row
    state: dict = field(default_factory=dict)  # model-specific cached tensors


class MMShapModel(ABC):
    """Interface a vision-and-language model implements to be probed by MM-SHAP."""

    key: str                  # short id used in result keys, e.g. "clip"
    title: str                # human-readable name for reports, e.g. "CLIP"
    reports_accuracy: bool    # True if the score is a probability (0.5-thresholdable)
    start_token: int          # text CLS/BOS id, never masked
    end_token: Optional[int]  # text SEP/EOS id to preserve (None if model has none)
    masked_batch_size: int = 64  # forward-pass batch; set by mm-shap.py's --batch-size

    @abstractmethod
    def encode_image(self, image_path: str):
        """Per-sample image work (e.g. loading, transforming, feature extraction)."""

    @abstractmethod
    def preprocess(self, image_ctx, sentence: str) -> Optional[Sample]:
        """Build a :class:`Sample` for one sentence, or ``None`` to skip it."""

    @abstractmethod
    def predict(self, sample: Sample) -> float:
        """Image-text matching score on the unmasked input."""

    @abstractmethod
    def predict_masked(self, sample: Sample, input_ids: torch.Tensor,
                       images: torch.Tensor) -> np.ndarray:
        """Matching scores for a batch of masked variants (masked text + patches)."""

    def score_variants(self, sample: Sample, features: np.ndarray) -> np.ndarray:
        """SHAP callback: score all masked rows (text ids then patch ids) in batches."""
        split = sample.input_ids.shape[1]
        input_ids = torch.tensor(features[:, :split])
        patch_ids = features[:, split:]
        scores = np.zeros(len(features))
        for start in range(0, len(features), self.masked_batch_size):
            rows = slice(start, start + self.masked_batch_size)
            images = torch.cat([
                zero_patches(sample.image.clone(), ids,
                             sample.patch_h, sample.patch_w, sample.grid_cols)
                for ids in patch_ids[rows]
            ])
            scores[rows] = self.predict_masked(sample, input_ids[rows], images)
        return scores


def mm_shap_score(model: MMShapModel, sample: Sample) -> float:
    """Textual contribution share for one (image, sentence) pair; see module doc."""
    p = patch_grid_size(sample.n_text)
    patch_ids = torch.tensor(range(1, p ** 2 + 1)).unsqueeze(0)
    features = torch.cat((sample.input_ids, patch_ids), 1).unsqueeze(1)

    def masker(mask, row):
        return mask_features(row, mask, model.start_token, model.end_token,
                             sample.n_text)

    def predict(rows):
        return model.score_variants(sample, rows)

    explainer = shap.Explainer(predict, masker, silent=True)
    contributions = np.abs(explainer(features).values[0, 0])

    text_contribution = contributions[:sample.n_text].sum()
    image_contribution = contributions[sample.n_text:].sum()
    return text_contribution / (text_contribution + image_contribution)
