"""Pre-download model weights into ./models_local so BOTH dependency stacks can
load them from a local path.

The legacy "before" env (transformers 4.11.1) cannot download from the current
HuggingFace Hub, so the regression harness relies on local snapshots. Loading from
a local directory also guarantees the two stacks use byte-identical weights.

Usage (inside the uv env):
    uv run python scripts/fetch_local_models.py                 # all
    uv run python scripts/fetch_local_models.py --only clip
"""
import argparse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "models_local"

# HuggingFace Hub snapshots; the local dir name is derived by replacing '/'.
HF_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "lxmert": "unc-nlp/lxmert-base-uncased",
    "bert": "bert-base-uncased",
}
# Faster R-CNN is served from the legacy S3 bucket (no live HF Hub repo).
FRCNN_ID = "unc-nlp/frcnn-vg-finetuned"
FRCNN_S3 = "https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned"
FRCNN_FILES = ("config.yaml", "pytorch_model.bin")


def local_dir(model_id: str) -> Path:
    return DEST / model_id.replace("/", "__")


def fetch_hf(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    target = local_dir(model_id)
    print(f"snapshot {model_id} -> {target}")
    snapshot_download(model_id, local_dir=str(target))


def download(url: str, dest: Path, timeout: int = 900) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)


def fetch_frcnn() -> None:
    target = local_dir(FRCNN_ID)
    for name in FRCNN_FILES:
        dest = target / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"frcnn {name}: already present")
            continue
        url = f"{FRCNN_S3}/{name}"
        print(f"frcnn {name} <- {url}")
        download(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[*HF_MODELS, "frcnn", "all"], default="all")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    todo = [*HF_MODELS, "frcnn"] if args.only == "all" else [args.only]
    for name in todo:
        if name == "frcnn":
            fetch_frcnn()
        else:
            fetch_hf(HF_MODELS[name])

    print("\nDone. models_local/ contents:")
    for entry in sorted(p.name for p in DEST.iterdir()):
        print("  ", entry)


if __name__ == "__main__":
    main()
