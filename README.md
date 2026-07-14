# MM-SHAP

This is the official implementation of the paper "MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks" accepted at ACL 2023 Toronto https://aclanthology.org/2023.acl-long.223/ .

## Cite
```
@inproceedings{parcalabescu-frank-2023-mm,
    title = "{MM}-{SHAP}: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models {\&} Tasks",
    author = "Parcalabescu, Letitia  and
      Frank, Anette",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.223",
    doi = "10.18653/v1/2023.acl-long.223",
    pages = "4032--4059",
    abstract = "Vision and language models (VL) are known to exploit unrobust indicators in individual modalities (e.g., introduced by distributional biases) instead of focusing on relevant information in each modality. That a unimodal model achieves similar accuracy on a VL task to a multimodal one, indicates that so-called unimodal collapse occurred. However, accuracy-based tests fail to detect e.g., when the model prediction is wrong, while the model used relevant information from a modality.Instead, we propose MM-SHAP, a performance-agnostic multimodality score based on Shapley values that reliably quantifies in which proportions a multimodal model uses individual modalities. We apply MM-SHAP in two ways: (1) to compare models for their average degree of multimodality, and (2) to measure for individual models the contribution of individual modalities for different tasks and datasets.Experiments with six VL models {--} LXMERT, CLIP and four ALBEF variants {--} on four VL tasks highlight that unimodal collapse can occur to different degrees and in different directions, contradicting the wide-spread assumption that unimodal collapse is one-sided. Based on our results, we recommend MM-SHAP for analysing multimodal tasks, to diagnose and guide progress towards multimodal integration. Code available at https://github.com/Heidelberg-NLP/MM-SHAP.",
}
```

## Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/). To create the environment (pinned to Python 3.10):

```bash
uv sync
```

This installs everything declared in `pyproject.toml` and pinned in `uv.lock` into a local `.venv/`. Run any script through uv so it uses that environment:

```bash
uv run python mm-shap.py <model> <num_samples> [--write]
```

### HuggingFace access
CLIP, LXMERT, Faster R-CNN and `bert-base-uncased` download from the HuggingFace Hub on
first run. To point downloads at a custom cache directory (and pass a token for gated
models), copy `.env.example` to `.env` and set `HF_CACHE` / `HF_TOKEN`; `mm-shap.py` loads
it at startup.

## Models
Three models are supported, selected via the first argument to `mm-shap.py`. CLIP and
LXMERT are downloaded automatically from the HuggingFace Hub on first run; ALBEF needs a
one-time setup.

| `model` arg | Model | Weights source |
| --- | --- | --- |
| `clip` | CLIP (`openai/clip-vit-base-patch32`) | HuggingFace Hub (automatic) |
| `lxmert` | LXMERT (`unc-nlp/lxmert-base-uncased`) + Faster R-CNN (`unc-nlp/frcnn-vg-finetuned`) | HuggingFace Hub / legacy S3 (automatic) |
| `albef` | ALBEF (finetuned checkpoints) | `scripts/setup_albef.py` |

The ALBEF model code is vendored in `ALBEF/` (from
[salesforce/ALBEF](https://github.com/salesforce/ALBEF); see `ALBEF/README.md`).
You only need to download a checkpoint:

```bash
uv run python scripts/setup_albef.py                 # default: flickr30k checkpoint
# other checkpoints: --checkpoint {flickr30k,mscoco,refcoco,vqa,ALBEF,ALBEF_4M,all}
```

Notes on modernization:
* The Faster R-CNN loader (`LXMERT/modeling_frcnn.py`) defaulted to the dead
  `cdn.huggingface.co`; it now uses the still-live legacy S3 bucket.
* ALBEF's vendored `xbert.py` used a removed `add_code_sample_docstrings(tokenizer_class=...)`
  argument (patched away), and the ALBEF adapter now uses the stock
  `transformers.BertTokenizer` (identical output for `bert-base-uncased`).

Model weights (LXMERT/CLIP from the HuggingFace Hub, ALBEF `.pth` checkpoints) and
datasets are gitignored. The ALBEF and LXMERT *code* is vendored (see the
third-party license note below).

## Dataset (foil-benchmark / VALSE)
Experiments use the VALSE benchmark ([Heidelberg-NLP/VALSE](https://github.com/Heidelberg-NLP/VALSE)).
`scripts/prepare_foil_sample.py` downloads a small, self-contained sample for one
instrument: it fetches the annotation json and a handful of images (for `existence`,
the Visual7W ids map to public Visual Genome images) into `data/foil-benchmark/`.

```bash
uv run python scripts/prepare_foil_sample.py --instrument existence --num 20
```

This writes `data/foil-benchmark/annotations/existence.sample.json` and images under
`data/foil-benchmark/images/existence/`. The `VALSE_DATA` dict in `mmshap/evaluation.py`
already points at this sample for the `existence` instrument.

## Usage
Run `mm-shap.py <model> <num_samples>` from the repository root (so that `import shap`
resolves to the vendored `shap/` package):

```bash
uv run python mm-shap.py clip 20                        # num_samples is an int or "all"
uv run python scripts/setup_albef.py                    # once, for ALBEF
uv run python mm-shap.py albef 20 --checkpoint flickr30k --write   # --write saves result jsons
```

Masked (image, sentence) variants are scored in batches; `--batch-size N` (default 64)
trades GPU memory for speed. `--write` saves per-sample result jsons under `result_jsons/`.

For the full benchmark, download the datasets from their sources
(VALSE 💃 https://github.com/Heidelberg-NLP/VALSE, VQA https://visualqa.org/download.html,
GQA https://cs.stanford.edu/people/dorarad/gqa/download.html) and add them to
`VALSE_DATA` in `mmshap/evaluation.py`.

## Credits & third-party licenses
This repository is MIT-licensed (see `LICENSE`), but it bundles third-party code
under its own license:

* The Shapley value implementation in the `shap` folder is a modified version of
  https://github.com/slundberg/shap (MIT).
* The `LXMERT/` folder (Faster R-CNN feature extraction) is from the HuggingFace
  `transformers` LXMERT example and is licensed under **Apache License 2.0**, not
  MIT. See `LXMERT/LICENSE` and `LXMERT/README.md` for provenance and the list of
  modifications.
* The `ALBEF/` folder is vendored from
  [salesforce/ALBEF](https://github.com/salesforce/ALBEF), licensed **BSD-3-Clause**
  (`ALBEF/LICENSE`), except `models/xbert.py` and `models/tokenization_bert.py`,
  which retain their HuggingFace-derived **Apache-2.0** headers. See
  `ALBEF/README.md` for provenance and modifications. Only the large `.pth`
  checkpoints are fetched at setup time (`scripts/setup_albef.py`).
