"""ALBEF adapter (finetuned image-text-matching checkpoints), run on GPU."""
import re
from functools import partial
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import BertTokenizer

from ALBEF.models.vit import VisionTransformer
from ALBEF.models.xbert import BertConfig, BertModel
from mmshap.core import MMShapModel, Sample
from mmshap.masking import patch_grid_size

IMAGE_SIZE = 384
BERT_CONFIG = "ALBEF/configs/config_bert.json"

_normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    _normalize,
])


class VLTransformerITM(nn.Module):
    """ALBEF's vision-language image-text-matching model (ViT + cross-modal BERT)."""

    def __init__(self, text_encoder: str, config_bert: str) -> None:
        super().__init__()
        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=IMAGE_SIZE, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False)
        self.itm_head = nn.Linear(768, 2)

    def forward(self, image: torch.Tensor, text) -> torch.Tensor:
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts, return_dict=True)
        return self.itm_head(output.last_hidden_state[:, 0, :])


def pre_caption(caption: str, max_words: int = 30) -> str:
    """ALBEF's text normalization: strip punctuation, collapse whitespace, truncate."""
    caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower())
    caption = caption.replace('-', ' ').replace('/', ' ')
    caption = re.sub(r"\s{2,}", ' ', caption).rstrip('\n').strip(' ')
    words = caption.split(' ')
    return ' '.join(words[:max_words]) if len(words) > max_words else caption


class Albef(MMShapModel):
    key = "albef"
    reports_accuracy = True
    start_token = 101
    end_token = None  # ALBEF does not reset a SEP token when masking

    def __init__(self, checkpoint: str) -> None:
        self.title = f"ALBEF {checkpoint}"
        bert = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        self.model = VLTransformerITM(text_encoder=bert, config_bert=BERT_CONFIG)
        state = torch.load(f"ALBEF/checkpoints/{checkpoint}.pth", map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model.cuda()

    def encode_image(self, image_path: str) -> torch.Tensor:
        return _transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    def preprocess(self, image_ctx: torch.Tensor, sentence: str) -> Optional[Sample]:
        text_input = self.tokenizer(pre_caption(sentence), return_tensors="pt")
        n_text = text_input.input_ids.shape[1]
        patch = IMAGE_SIZE // patch_grid_size(n_text)
        return Sample(
            input_ids=text_input.input_ids,
            n_text=n_text,
            image=image_ctx,
            patch_h=patch,
            patch_w=patch,
            grid_cols=IMAGE_SIZE // patch,
            state={"text_input": text_input},
        )

    def predict(self, sample: Sample) -> float:
        text = sample.state["text_input"].copy().to("cuda")
        return self._match(sample.image.cuda(), text)[0].item()

    def predict_masked(self, sample: Sample, input_ids: torch.Tensor,
                       images: torch.Tensor) -> np.ndarray:
        text = sample.state["text_input"].copy()
        text["input_ids"] = input_ids
        text["attention_mask"] = torch.ones_like(input_ids)
        with torch.no_grad():
            return self._match(images.cuda(), text.to("cuda")).numpy()

    def _match(self, images: torch.Tensor, text) -> torch.Tensor:
        output = self.model(images, text)
        return torch.nn.Softmax(dim=1)(output).cpu().detach()[:, 1]
