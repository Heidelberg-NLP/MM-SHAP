"""Download an ALBEF checkpoint.

The ALBEF model code is vendored in the `ALBEF/` directory (see ALBEF/README.md
for provenance and license). This script only downloads the finetuned
checkpoints, which are too large to vendor.

Usage (inside the uv env):
    uv run python scripts/setup_albef.py                 # default: flickr30k checkpoint
    uv run python scripts/setup_albef.py --checkpoint mscoco
    uv run python scripts/setup_albef.py --checkpoint all
"""
import argparse
import os

CKPT_BASE = "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF"
CHECKPOINTS = ("flickr30k", "mscoco", "refcoco", "vqa", "ALBEF", "ALBEF_4M")


def download(url, dest, timeout=900):
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="flickr30k",
                    help="which checkpoint(s) to download: one of "
                         f"{CHECKPOINTS} or 'all'")
    ap.add_argument("--root", default="ALBEF")
    args = ap.parse_args()

    which = CHECKPOINTS if args.checkpoint == "all" else [args.checkpoint]
    for name in which:
        if name not in CHECKPOINTS:
            raise SystemExit(f"unknown checkpoint {name!r}; choose from {CHECKPOINTS} or 'all'")
        dest = os.path.join(args.root, "checkpoints", f"{name}.pth")
        if os.path.exists(dest):
            print(f"checkpoint {name}.pth already present, skipping")
            continue
        print(f"Downloading checkpoint {name}.pth ...")
        n = download(f"{CKPT_BASE}/{name}.pth", dest)
        print(f"  {name}.pth: {n/1e6:.0f} MB")

    print("\nALBEF checkpoint setup complete.")


if __name__ == "__main__":
    main()
