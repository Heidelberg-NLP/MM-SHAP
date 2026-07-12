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

> The legacy conda files (`environment.yml`, `requirements_conda.txt`, `requirements_pip.txt`) describe the original Python 3.6 stack and are kept for reference only.

### Legacy "before" environment (for regression testing)
To reproduce the original Python 3.6 stack (e.g. to generate baseline outputs and check the modernized code matches), a minimal, runnable conda spec is provided in `environment.before.yml`. It is built with a local [micromamba](https://mamba.readthedocs.io/):

```bash
# one-time: download the micromamba binary into ./bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# create the env (into ./.micromamba, both gitignored)
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
./bin/micromamba env create -y -f environment.before.yml

# run in the legacy env
./bin/micromamba run -n shap-before python mm-shap.py <model> <num_samples> [--write]
```

This installs the original pins (Python 3.6.13, `torch 1.9.1`/CUDA 11.1, `transformers 4.11.1`, `numpy 1.19.2`, ...). Notebook/experiment tooling from the original file is intentionally omitted.

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

Notes on modernization (kept identical between the "before" and "after" stacks):
* The Faster R-CNN loader (`LXMERT/modeling_frcnn.py`) defaulted to the dead
  `cdn.huggingface.co`; it now uses the still-live legacy S3 bucket.
* ALBEF's vendored `xbert.py` used a removed `add_code_sample_docstrings(tokenizer_class=...)`
  argument (patched away), and the ALBEF script now uses the stock
  `transformers.BertTokenizer` (identical output for `bert-base-uncased`).

Model weights (LXMERT/CLIP from the HuggingFace Hub, ALBEF `.pth` checkpoints) and
datasets are gitignored. The ALBEF and LXMERT *code* is vendored (see the
third-party license note below).

### Local model snapshots (needed for the legacy env)
The legacy `transformers 4.11.1` in the "before" env cannot download from the
current HuggingFace Hub. `scripts/fetch_local_models.py` pre-downloads the model
files into `models_local/` (gitignored) so **both** stacks load from a local path
(also guaranteeing byte-identical weights):

```bash
uv run python scripts/fetch_local_models.py          # CLIP, LXMERT, BERT, Faster R-CNN
```

At runtime the scripts call `mmshap_repro.resolve_model(<hf id>)`, which returns the
local snapshot if present and otherwise falls back to the normal Hub id, so the
`.venv` ("after") env still works without this step.

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

For the full benchmark, download the datasets from their sources
(VALSE 💃 https://github.com/Heidelberg-NLP/VALSE, VQA https://visualqa.org/download.html,
GQA https://cs.stanford.edu/people/dorarad/gqa/download.html) and add them to
`VALSE_DATA` in `mmshap/evaluation.py`.

## Regression testing (before vs. after)
To check that the modernized (uv / Python 3.10) stack reproduces the legacy
(conda / Python 3.6) stack, `scripts/regression_test.py` runs the *same* fixed
sample through both environments and compares the per-sample model predictions and
t-SHAP scores. Both stacks share the same vendored `shap/` source, so this isolates
the effect of the dependency upgrade (torch 1.9→2.2, transformers 4.11→4.39,
numpy 1.19→1.26).

Prerequisites: create both envs (`uv sync` and the `environment.before.yml` env
above), fetch the local model snapshots, and prepare the data sample.

```bash
uv run python scripts/fetch_local_models.py
uv run python scripts/prepare_foil_sample.py --instrument existence --num 20

uv run python scripts/regression_test.py --model clip   --num 3
uv run python scripts/regression_test.py --model lxmert --num 3
uv run python scripts/regression_test.py --model albef  --num 3 --checkpoint flickr30k
```

Determinism is controlled by the `MMSHAP_SEED` environment variable (set by the
harness; unset means the original stochastic behaviour). When set, `mmshap_repro.py`
seeds Python/NumPy/torch so the shap permutation orderings match across stacks.

Expected parity (numbers from the `existence` sample, 3 samples, seed 0):
* **CLIP** (CPU) reproduces almost exactly — max |Δ| ≈ 2e-3.
* **ALBEF** (GPU, ViT + BERT) also reproduces tightly — max |Δ| ≈ 4e-4.
* **LXMERT** (GPU, incl. Faster R-CNN feature extraction) is looser — max |Δ| ≈ 8e-2.
  The RoI-pooling / NMS ops in FRCNN differ across CUDA/torch versions, so LXMERT
  reproduces only *qualitatively* (a few percent at the logit level, but identical
  pairwise accuracy).

The harness uses a per-model default tolerance (CLIP `1e-2`, LXMERT/ALBEF `1e-1`);
pass `--tol` to override.

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
