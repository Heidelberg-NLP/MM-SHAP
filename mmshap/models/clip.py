"""CLIP adapter (openai/clip-vit-base-patch32)."""
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mmshap.core import MMShapModel, Sample
from mmshap.masking import patch_grid_size

IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Clip(MMShapModel):
    key = "clip"
    title = "CLIP"
    reports_accuracy = False  # logits_per_image is a raw similarity, not a probability
    start_token = 49406
    end_token = 49407

    def __init__(self) -> None:
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(name)

    def encode_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)  # processor couples text+image (see preprocess)

    def preprocess(self, image_ctx: Image.Image, sentence: str) -> Optional[Sample]:
        try:
            inputs = self.processor(text=sentence, images=image_ctx,
                                    return_tensors="pt", padding=True)
        except Exception:  # image feature extraction can fail on some files
            return None
        n_text = inputs.input_ids.shape[1]
        patch = IMAGE_SIZE // patch_grid_size(n_text)
        return Sample(
            input_ids=inputs.input_ids,
            n_text=n_text,
            image=inputs.pixel_values,
            patch_h=patch,
            patch_w=patch,
            grid_cols=IMAGE_SIZE // patch,
            state={"attention_mask": inputs.attention_mask},
        )

    def predict(self, sample: Sample) -> float:
        out = self.model(input_ids=sample.input_ids.to(DEVICE),
                         attention_mask=sample.state["attention_mask"].to(DEVICE),
                         pixel_values=sample.image.to(DEVICE))
        return out.logits_per_image[0, 0].item()

    def predict_masked(self, sample: Sample, input_ids: torch.Tensor,
                       images: torch.Tensor) -> np.ndarray:
        attention_mask = sample.state["attention_mask"].repeat(len(input_ids), 1)
        with torch.no_grad():
            out = self.model(input_ids=input_ids.to(DEVICE),
                             attention_mask=attention_mask.to(DEVICE),
                             pixel_values=images.to(DEVICE))
        # each masked image is paired with its own masked text: take the diagonal.
        return out.logits_per_image.diagonal().detach().cpu().numpy()
