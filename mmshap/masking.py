"""Feature masking for MM-SHAP.

The SHAP features of an (image, sentence) pair are its text tokens followed by a grid
of image patches. This module turns SHAP's on/off masks into concrete masked inputs:
text token ids get zeroed, and masked image-patch ids (encoded as 0) tell the model
wrapper which patches to black out.
"""
import math
from typing import Optional

import numpy as np
import torch


def patch_grid_size(n_text_tokens: int) -> int:
    """Side length p of the p×p image-patch grid, chosen so the image contributes about
    as many features as the text (p = ceil(sqrt(number of text tokens)))."""
    return int(math.ceil(np.sqrt(n_text_tokens)))


def mask_features(features: torch.Tensor, keep: np.ndarray, start_token: int,
                  end_token: Optional[int], n_text: int) -> torch.Tensor:
    """Zero the token/patch ids that SHAP masks out, always preserving the text
    sequence's special tokens (a zeroed patch id later means "black out this patch")."""
    masked = features.clone()
    keep = torch.tensor(keep).unsqueeze(0)
    masked[~keep] = 0
    masked[0, 0] = start_token
    if end_token is not None:
        masked[0, n_text - 1] = end_token
    return masked


def zero_patches(image: torch.Tensor, patch_ids: np.ndarray, patch_h: int, patch_w: int,
                 grid_cols: int) -> torch.Tensor:
    """Black out (in place) the image patches whose id was masked to 0."""
    for k, patch_id in enumerate(patch_ids):
        if patch_id == 0:
            row, col = k // grid_cols, k % grid_cols
            image[:, :, row * patch_h:(row + 1) * patch_h,
                  col * patch_w:(col + 1) * patch_w] = 0
    return image
