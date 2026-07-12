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

## Usage
To run experiments with CLIP, LXMERT and ALBEF models, run the corresponding script `mm-shap_[MODEL]_dataset.py`. You need to download the data from their corresponding repositories, for example:
* VALSE 💃: https://github.com/Heidelberg-NLP/VALSE
* VQA: https://visualqa.org/download.html
* GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .
