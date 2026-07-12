"""LXMERT adapter (unc-nlp/lxmert-base-uncased), run on GPU.

LXMERT consumes region features rather than raw pixels, so masking an image patch means
re-running the Faster R-CNN feature extractor on the patched image before LXMERT.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The vendored LXMERT modules import `utils` (LXMERT/utils.py) as a top-level module,
# so make LXMERT/ importable regardless of the current working directory.
_LXMERT_DIR = Path(__file__).resolve().parents[2] / "LXMERT"
sys.path.insert(0, str(_LXMERT_DIR))

from transformers import LxmertForPreTraining, LxmertTokenizer  # noqa: E402
from utils import Config  # noqa: E402

from LXMERT.modeling_frcnn import GeneralizedRCNN  # noqa: E402
from LXMERT.processing_image import Preprocess  # noqa: E402
from mmshap.core import MMShapModel, Sample  # noqa: E402
from mmshap.masking import patch_grid_size  # noqa: E402
from mmshap_repro import resolve_model  # noqa: E402


class Lxmert(MMShapModel):
    key = "lxmert"
    title = "LXMERT"
    reports_accuracy = True
    start_token = 101
    end_token = 102

    def __init__(self) -> None:
        frcnn_id = resolve_model("unc-nlp/frcnn-vg-finetuned")
        lxmert_id = resolve_model("unc-nlp/lxmert-base-uncased")
        self.frcnn_cfg = Config.from_pretrained(frcnn_id)
        self.frcnn_cfg.MODEL.DEVICE = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained(
            frcnn_id, config=self.frcnn_cfg).cuda()
        self.image_preprocess = Preprocess(self.frcnn_cfg)
        self.tokenizer = LxmertTokenizer.from_pretrained(lxmert_id)
        self.model = LxmertForPreTraining.from_pretrained(lxmert_id).cuda()

    def encode_image(self, image_path: str) -> dict:
        images, sizes, scales_yx = self.image_preprocess(image_path)
        features, boxes = self._extract_features(images, sizes, scales_yx)
        return {"images": images, "sizes": sizes, "scales_yx": scales_yx,
                "features": features, "boxes": boxes}

    def preprocess(self, image_ctx: dict, sentence: str) -> Optional[Sample]:
        inputs = self.tokenizer(sentence, padding=False, truncation=True,
                                return_token_type_ids=True, return_attention_mask=True,
                                add_special_tokens=True, return_tensors="pt")
        n_text = int(np.count_nonzero(inputs.input_ids))
        images = image_ctx["images"]
        p = patch_grid_size(n_text)
        return Sample(
            input_ids=inputs.input_ids,
            n_text=n_text,
            image=images,
            patch_h=images.shape[2] // p,
            patch_w=images.shape[3] // p,
            grid_cols=p,
            state={"ctx": image_ctx, "attention_mask": inputs.attention_mask,
                   "token_type_ids": inputs.token_type_ids},
        )

    def predict(self, sample: Sample) -> float:
        ctx = sample.state["ctx"]
        return self._match_score(sample, sample.input_ids,
                                 ctx["features"], ctx["boxes"])

    def predict_masked(self, sample: Sample, input_ids: torch.Tensor,
                       image: torch.Tensor) -> float:
        ctx = sample.state["ctx"]
        features, boxes = self._extract_features(image, ctx["sizes"], ctx["scales_yx"])
        return self._match_score(sample, input_ids, features, boxes)

    def _extract_features(self, images: torch.Tensor, sizes, scales_yx):
        out = self.frcnn(images.cuda(), sizes, scales_yx=scales_yx,
                         padding="max_detections",
                         max_detections=self.frcnn_cfg.max_detections,
                         return_tensors="pt")
        return out.get("roi_features"), out.get("normalized_boxes")

    def _match_score(self, sample: Sample, input_ids: torch.Tensor,
                     features: torch.Tensor, boxes: torch.Tensor) -> float:
        out = self.model(
            input_ids=input_ids.cuda(),
            attention_mask=sample.state["attention_mask"].cuda(),
            visual_feats=features.cuda(),
            visual_pos=boxes.cuda(),
            token_type_ids=sample.state["token_type_ids"].cuda(),
            return_dict=True,
            output_attentions=False,
        )
        scores = torch.nn.Softmax(dim=1)(out["cross_relationship_score"])
        return scores.cpu().detach()[:, 1].item()
