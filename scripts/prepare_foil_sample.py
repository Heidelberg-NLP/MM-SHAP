"""Prepare a small, self-contained sample of the VALSE "foil-benchmark" dataset.

For a given VALSE instrument (default: ``existence``) this script downloads the
annotation json from the VALSE repo, selects the first ``--num`` entries, fetches
their images, and writes a reduced annotation json containing only those entries.

The ``existence`` instrument uses Visual7W images (``v7w_<id>.jpg``); the numeric
id is a Visual Genome image id, so images are fetched from the public Visual Genome
mirror (``VG_100K`` / ``VG_100K_2``) and saved under the Visual7W name.

A ``foils`` list field is added (mirroring the single ``foil`` string) so the
``mm-shap_*`` scripts, which expect ``foil["foils"][0]``, run unmodified.

Usage (inside the uv env):
    uv run python scripts/prepare_foil_sample.py --instrument existence --num 20
"""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

VALSE_RAW = "https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/main/data"
VG_MIRRORS = (
    "https://cs.stanford.edu/people/rak248/VG_100K",
    "https://cs.stanford.edu/people/rak248/VG_100K_2",
)
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


def download(url: str, dest: Path, timeout: int = 120) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    dest.write_bytes(data)
    return len(data)


def fetch_vg_image(vg_id: str, dest: Path, timeout: int = 60) -> Optional[str]:
    for base in VG_MIRRORS:
        url = f"{base}/{vg_id}.jpg"
        try:
            download(url, dest, timeout=timeout)
            return url
        except (urllib.error.URLError, TimeoutError):
            continue
    return None


def with_foils(entry: dict) -> dict:
    entry = dict(entry)
    if "foils" not in entry and "foil" in entry:
        entry["foils"] = [entry["foil"]]
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="existence", choices=sorted(INSTRUMENT_FILES))
    ap.add_argument("--num", type=int, default=20, help="number of samples to keep")
    ap.add_argument("--root", default="data/foil-benchmark")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = root / "annotations"
    img_dir = root / "images" / args.instrument
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    ann_path = ann_dir / INSTRUMENT_FILES[args.instrument]
    if not ann_path.exists():
        url = f"{VALSE_RAW}/{INSTRUMENT_FILES[args.instrument]}"
        print(f"Downloading annotations: {url}")
        download(url, ann_path)
    full = json.loads(ann_path.read_text())
    print(f"Full annotation set: {len(full)} entries")

    sample: dict[str, dict] = {}
    for key, entry in full.items():
        if len(sample) >= args.num:
            break
        image_file = entry["image_file"]  # e.g. v7w_2371044.jpg
        dest = img_dir / image_file
        if not dest.exists():
            url = fetch_vg_image(entry["dataset_idx"], dest)
            if url is None:
                print(f"  skip {image_file}: image not found on any mirror")
                continue
            print(f"  {image_file}: {dest.stat().st_size} bytes")
        sample[key] = with_foils(entry)

    out_path = ann_dir / f"{args.instrument}.sample.json"
    out_path.write_text(json.dumps(sample, indent=2))
    print(f"\nWrote {len(sample)} samples -> {out_path}")
    print(f"Images -> {img_dir}")


if __name__ == "__main__":
    main()
