# LXMERT / Faster R-CNN feature extraction code

The files in this directory (`modeling_frcnn.py`, `utils.py`, `processing_image.py`)
are third-party code, **not** covered by the MIT license in the repository root.

- **Origin:** the LXMERT example in HuggingFace `transformers`
  (https://github.com/huggingface/transformers/tree/main/examples/research_projects/lxmert),
  originally by Antonio Mendoza, Hao Tan, Mohit Bansal, adapted from
  Facebook Inc / Detectron2.
- **License:** Apache License 2.0 (full text in [`LICENSE`](./LICENSE); the
  per-file headers also reference it).
- **Modifications:** `modeling_frcnn.py` has been modified for MM-SHAP — the
  default weight source was changed from the dead CloudFront CDN
  (`cdn.huggingface.co`) to the legacy S3 bucket (`use_cdn` defaults to `False`).
  `utils.py` and `processing_image.py` are unmodified.
