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
uv run python mm-shap_clip_dataset.py <num_samples> <write_res>
```

> The legacy conda files (`environment.yml`, `requirements_conda.txt`, `requirements_pip.txt`) describe the original Python 3.6 stack and are kept for reference only.

### Legacy "before" environment (for regression testing)
To reproduce the original Python 3.6 stack (e.g. to generate baseline outputs and check the modernized code matches), a minimal, runnable conda spec is provided in `environment.before.yml`. It is built with a local [micromamba](https://mamba.readthedocs.io/):

```bash
# one-time: download the micromamba binary into ./bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# create the env (into ./.micromamba, both gitignored)
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
./bin/micromamba env create -y -f environment.before.yml

# run a script in the legacy env
./bin/micromamba run -n shap-before python mm-shap_clip_dataset.py <num_samples> <write_res>
```

This installs the original pins (Python 3.6.13, `torch 1.9.1`/CUDA 11.1, `transformers 4.11.1`, `numpy 1.19.2`, ...). Notebook/experiment tooling from the original file is intentionally omitted.

## Models
Three models are supported, one per script. CLIP and LXMERT are downloaded
automatically from the HuggingFace Hub on first run; ALBEF needs a one-time setup.

| Script | Model | Weights source |
| --- | --- | --- |
| `mm-shap_clip_dataset.py` | CLIP (`openai/clip-vit-base-patch32`) | HuggingFace Hub (automatic) |
| `mm-shap_lxmert_dataset.py` | LXMERT (`unc-nlp/lxmert-base-uncased`) + Faster R-CNN (`unc-nlp/frcnn-vg-finetuned`) | HuggingFace Hub / legacy S3 (automatic) |
| `mm-shap_albef_dataset.py` | ALBEF (finetuned checkpoints) | `scripts/setup_albef.py` |

The ALBEF model code is vendored in `ALBEF/` (from
[salesforce/ALBEF](https://github.com/salesforce/ALBEF); see `ALBEF/README.md`).
You only need to download a checkpoint:

```bash
uv run python scripts/setup_albef.py                 # default: flickr30k checkpoint
# other checkpoints: --checkpoint {flickr30k,mscoco,refcoco,vqa,ALBEF,ALBEF_4M,all}
```

Notes on modernization (kept identical between the "before" and "after" stacks):
* The Faster R-CNN loader (`LXMERT/modeling_frcnn.py`) defaulted to the dead
  `cdn.huggingface.co`; it now uses the still-live legacy S3 bucket.
* ALBEF's vendored `xbert.py` used a removed `add_code_sample_docstrings(tokenizer_class=...)`
  argument (patched away), and the ALBEF script now uses the stock
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
`data/foil-benchmark/images/existence/`. The `DATA` dict in each `mm-shap_*` script
already points at this sample for the `existence` instrument.

## Usage
Run the corresponding script `mm-shap_[MODEL]_dataset.py` from the repository root
(so that `import shap` resolves to the vendored `shap/` package):

```bash
uv run python mm-shap_clip_dataset.py 20 no      # <num_samples: int|"all"> <write_res: yes|no>
uv run python scripts/setup_albef.py             # once, for ALBEF
uv run python mm-shap_albef_dataset.py 20 flickr30k no   # <num_samples> <checkpoint> <write_res>
```

For the full benchmark, download the datasets from their sources
(VALSE 💃 https://github.com/Heidelberg-NLP/VALSE, VQA https://visualqa.org/download.html,
GQA https://cs.stanford.edu/people/dorarad/gqa/download.html) and adjust the `DATA` dict.

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
