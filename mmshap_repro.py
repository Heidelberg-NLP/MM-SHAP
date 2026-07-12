"""Deterministic seeding and local-snapshot resolution shared by the mm-shap scripts.

Setting the ``MMSHAP_SEED`` environment variable makes a run reproducible: shap's
permutation explainer otherwise shuffles feature orderings with the global NumPy
RNG, so the before/after regression harness sets it to compare the two stacks on
equal footing. Unset means the original stochastic behaviour.

``resolve_model`` lets the legacy "before" env load models from a local snapshot,
since its old ``transformers`` cannot download from the current HuggingFace Hub.

This module is imported by both dependency stacks, so it stays Python 3.6-compatible.
"""
import os
from pathlib import Path
from typing import Optional

MODELS_LOCAL = Path(__file__).resolve().parent / "models_local"


def resolve_model(model_id: str) -> str:
    local = MODELS_LOCAL / model_id.replace("/", "__")
    return str(local) if local.is_dir() else model_id


def maybe_seed() -> Optional[int]:
    raw = os.environ.get("MMSHAP_SEED")
    if not raw:
        return None
    seed = int(raw)

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
