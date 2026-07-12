"""CLIP adapter (openai/clip-vit-base-patch32), run on CPU."""
import copy
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mmshap.core import MMShapModel, Sample
from mmshap.masking import patch_grid_size
from mmshap_repro import resolve_model

IMAGE_SIZE = 224


class Clip(MMShapModel):
    key = "clip"
    title = "CLIP"
    reports_accuracy = False  # logits_per_image is a raw similarity, not a probability
    start_token = 49406
    end_token = 49407

    def __init__(self) -> None:
        name = resolve_model("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained(name)
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
            state={"inputs": inputs},
        )

    def predict(self, sample: Sample) -> float:
        return self.model(**sample.state["inputs"]).logits_per_image[0, 0].item()

    def predict_masked(self, sample: Sample, input_ids: torch.Tensor,
                       image: torch.Tensor) -> float:
        inputs = copy.deepcopy(sample.state["inputs"])
        inputs["input_ids"] = input_ids
        inputs["pixel_values"] = image
        with torch.no_grad():
            return self.model(**inputs).logits_per_image.item()
