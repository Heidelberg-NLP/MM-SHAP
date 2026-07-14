"""Download an ALBEF checkpoint.

The ALBEF model code is vendored in the `ALBEF/` directory (see ALBEF/README.md
for provenance and license). This script only downloads the finetuned checkpoints,
which are too large to vendor.

Usage (inside the uv env):
    uv run python scripts/setup_albef.py                 # default: flickr30k checkpoint
    uv run python scripts/setup_albef.py --checkpoint mscoco
    uv run python scripts/setup_albef.py --checkpoint all
"""
import argparse
import urllib.request
from pathlib import Path

CKPT_BASE = "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF"
CHECKPOINTS = ("flickr30k", "mscoco", "refcoco", "vqa", "ALBEF", "ALBEF_4M")


def download(url: str, dest: Path, timeout: int = 900) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            total += len(chunk)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="flickr30k",
                    help=f"which checkpoint(s) to download: one of {CHECKPOINTS} or 'all'")
    ap.add_argument("--root", default="ALBEF")
    args = ap.parse_args()

    which = CHECKPOINTS if args.checkpoint == "all" else [args.checkpoint]
    for name in which:
        if name not in CHECKPOINTS:
            raise SystemExit(f"unknown checkpoint {name!r}; choose from {CHECKPOINTS} or 'all'")
        dest = Path(args.root) / "checkpoints" / f"{name}.pth"
        if dest.exists():
            print(f"checkpoint {name}.pth already present, skipping")
            continue
        print(f"Downloading checkpoint {name}.pth ...")
        size = download(f"{CKPT_BASE}/{name}.pth", dest)
        print(f"  {name}.pth: {size / 1e6:.0f} MB")

    print("\nALBEF checkpoint setup complete.")


if __name__ == "__main__":
    main()
