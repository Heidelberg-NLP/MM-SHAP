"""Verify a code change (e.g. a refactor) preserves MM-SHAP outputs.

Unlike ``regression_test.py`` (which compares the two *dependency stacks* running
the same code), this compares the *current code* against a golden baseline captured
earlier, in the same uv env with the same seed. A faithful refactor should reproduce
the baseline essentially bit-for-bit.

Workflow:
    # BEFORE changing code, capture the baseline:
    uv run python scripts/refactor_check.py --model clip --capture
    # AFTER changing code, check it still matches:
    uv run python scripts/refactor_check.py --model clip --check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_test import compare, run_stack

GOLDEN = Path(__file__).resolve().parent.parent / "regression_results" / "golden"


def golden_path(model: str, checkpoint: str) -> Path:
    name = f"albef_{checkpoint}" if model == "albef" else model
    return GOLDEN / f"{name}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip", "lxmert", "albef"])
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--checkpoint", default="flickr30k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="save the current output as the golden baseline")
    mode.add_argument("--check", action="store_true",
                      help="compare the current output against the golden baseline")
    args = ap.parse_args()

    result = run_stack("after", args.model, args.num, args.checkpoint, args.seed)
    dest = golden_path(args.model, args.checkpoint)

    if args.capture:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result))
        print(f"\ncaptured golden baseline -> {dest}")
        return

    if not dest.exists():
        raise SystemExit(f"no golden baseline at {dest}; run with --capture first")
    ok = compare(json.loads(dest.read_text()), result, args.tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
