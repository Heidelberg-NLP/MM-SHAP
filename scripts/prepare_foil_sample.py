"""Prepare a small, self-contained sample of the VALSE "foil-benchmark" dataset.

For a given VALSE instrument (default: ``existence``) this script:
  1. downloads the annotation json from the VALSE repo (if not present),
  2. selects the first ``--num`` entries,
  3. downloads their images to a local folder,
  4. writes a reduced annotation json containing only those entries.

The ``existence`` instrument uses Visual7W images (``v7w_<id>.jpg``); the numeric
id is a Visual Genome image id, so images are fetched from the public Visual
Genome mirror (``VG_100K`` / ``VG_100K_2``) and saved under the Visual7W name.

A ``foils`` list field is added (mirroring the single ``foil`` string) so the
``mm-shap_*`` scripts, which expect ``foil["foils"][0]``, run unmodified.

Usage (inside the uv env):
    uv run python scripts/prepare_foil_sample.py --instrument existence --num 20
"""
import argparse
import json
import os
import urllib.request

VALSE_RAW = "https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/main/data"
VG_MIRRORS = (
    "https://cs.stanford.edu/people/rak248/VG_100K",
    "https://cs.stanford.edu/people/rak248/VG_100K_2",
)
# VALSE instrument name -> annotation file name in the VALSE repo.
INSTRUMENT_FILES = {
    "existence": "existence.json",
    "plurals": "plurals.json",
    "counting-hard": "counting-hard.json",
    "counting-small-quant": "counting-small-quant.json",
    "counting-adversarial": "counting-adversarial.json",
    "relations": "relations.json",
    "action-replacement": "action-replacement.json",
    "actant-swap": "actant-swap.json",
    "coreference-standard": "coreference-standard.json",
    "coreference-hard": "coreference-hard.json",
    "foil-it": "foil-it.json",
}


def download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def fetch_vg_image(vg_id, dest, timeout=60):
    """Fetch a Visual Genome image by id, trying both mirror folders."""
    for base in VG_MIRRORS:
        url = f"{base}/{vg_id}.jpg"
        try:
            n = download(url, dest, timeout=timeout)
            return url, n
        except Exception:
            continue
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="existence", choices=sorted(INSTRUMENT_FILES))
    ap.add_argument("--num", type=int, default=20, help="number of samples to keep")
    ap.add_argument("--root", default="data/foil-benchmark")
    args = ap.parse_args()

    root = args.root
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "images", args.instrument)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    ann_path = os.path.join(ann_dir, INSTRUMENT_FILES[args.instrument])
    if not os.path.exists(ann_path):
        url = f"{VALSE_RAW}/{INSTRUMENT_FILES[args.instrument]}"
        print(f"Downloading annotations: {url}")
        download(url, ann_path)
    full = json.load(open(ann_path))
    print(f"Full annotation set: {len(full)} entries")

    sample = {}
    for key, entry in full.items():
        if len(sample) >= args.num:
            break
        image_file = entry["image_file"]  # e.g. v7w_2371044.jpg
        vg_id = entry["dataset_idx"]
        dest = os.path.join(img_dir, image_file)
        if not os.path.exists(dest):
            src, n = fetch_vg_image(vg_id, dest)
            if src is None:
                print(f"  skip {image_file}: image not found on any mirror")
                continue
            print(f"  {image_file}: {n} bytes")
        # normalize: add a `foils` list so mm-shap scripts run unmodified
        entry = dict(entry)
        if "foils" not in entry and "foil" in entry:
            entry["foils"] = [entry["foil"]]
        sample[key] = entry

    out_path = os.path.join(ann_dir, f"{args.instrument}.sample.json")
    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"\nWrote {len(sample)} samples -> {out_path}")
    print(f"Images -> {img_dir}")


if __name__ == "__main__":
    main()
