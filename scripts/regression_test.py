"""Before/after regression harness for MM-SHAP.

Runs the same fixed sample through both dependency stacks:
  * "before" : the legacy Python 3.6 conda env (`shap-before`, via micromamba)
  * "after"  : the modernized Python 3.10 uv env (`.venv`)

Both stacks share the *same* vendored `shap/` source, so this isolates the effect
of the dependency upgrade (torch 1.9->2.2, transformers 4.11->4.39, numpy 1.19->1.26).
Runs are made deterministic via MMSHAP_SEED so the per-sample model predictions and
t-SHAP scores can be compared directly.

Usage:
    uv run python scripts/regression_test.py --model clip --num 3
    uv run python scripts/regression_test.py --model albef --num 3 --checkpoint flickr30k
    uv run python scripts/regression_test.py --model clip --num 3 --env after   # single stack
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MICROMAMBA = REPO / "bin" / "micromamba"
MAMBA_ROOT = REPO / ".micromamba"
BEFORE_ENV = "shap-before"
INSTRUMENT = "existence"  # active DATA key in the scripts

# Per-model default tolerance on |before - after|. CLIP runs on CPU and reproduces
# almost exactly; LXMERT/ALBEF run on GPU (incl. FRCNN feature extraction), where
# CUDA-kernel/version differences (cuda 11.1->12.x, torch 1.9->2.2) are unavoidable,
# so we only expect qualitative parity (a few percent at the logit level).
MODEL_TOL = {"clip": 1e-2, "lxmert": 1e-1, "albef": 1e-1}


def result_json_path(model: str, num: int, checkpoint: str) -> Path:
    stem = f"albef_{checkpoint}_{num}" if model == "albef" else f"{model}_{num}"
    return REPO / "result_jsons" / stem / f"{INSTRUMENT}.json"


def script_args(model: str, num: int, checkpoint: str) -> list[str]:
    script = f"mm-shap_{model}_dataset.py"
    if model == "albef":
        return [script, str(num), checkpoint, "yes"]
    return [script, str(num), "yes"]


def run_stack(stack: str, model: str, num: int, checkpoint: str, seed: int) -> dict:
    args = script_args(model, num, checkpoint)
    env = dict(os.environ, MMSHAP_SEED=str(seed))
    if stack == "before":
        env["MAMBA_ROOT_PREFIX"] = str(MAMBA_ROOT)
        cmd = [str(MICROMAMBA), "run", "-n", BEFORE_ENV, "python", *args]
    else:
        cmd = ["uv", "run", "python", *args]

    out_path = result_json_path(model, num, checkpoint)
    out_path.unlink(missing_ok=True)

    print(f"\n=== [{stack}] {' '.join(cmd)} ===", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    summary = [
        line for line in proc.stdout.splitlines()
        if any(tag in line for tag in ("MM_score", "pairwise_accuracy", "tested"))
    ]
    print("\n".join(summary) if summary else proc.stdout[-500:])
    if proc.returncode != 0:
        print(f"[{stack}] STDERR tail:\n{proc.stderr[-1500:]}", file=sys.stderr)
        raise SystemExit(f"[{stack}] run failed (exit {proc.returncode})")
    if not out_path.exists():
        raise SystemExit(f"[{stack}] produced no result json at {out_path}")

    saved = REPO / "regression_results" / stack / f"{model}_{INSTRUMENT}.json"
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, saved)
    return json.loads(out_path.read_text())


def numeric_fields(entry: dict) -> dict[str, float]:
    return {
        k: v for k, v in entry.items()
        if isinstance(v, (int, float)) and (k.endswith("_model_prediction") or k.endswith("_t_shap"))
    }


def collect_diffs(before: dict, after: dict, keys: list[str]) -> dict[str, list[float]]:
    per_field: dict[str, list[float]] = {}
    for key in keys:
        bf, af = numeric_fields(before[key]), numeric_fields(after[key])
        for field, value in bf.items():
            if field in af:
                per_field.setdefault(field, []).append(abs(value - af[field]))
    return per_field


def compare(before: dict, after: dict, tol: float) -> bool:
    shared = [k for k in before if k in after]
    per_field = collect_diffs(before, after, shared)
    print("\n================ comparison ================")
    print(f"samples compared: {len(shared)}")
    if not per_field:
        raise SystemExit("no comparable numeric fields found (did both runs write results?)")

    worst = 0.0
    for field in sorted(per_field):
        diffs = per_field[field]
        mx = max(diffs)
        mean = sum(diffs) / len(diffs)
        worst = max(worst, mx)
        flag = "OK " if mx <= tol else "!! "
        print(f"  {flag}{field:34s} n={len(diffs):3d}  max|Δ|={mx:.3e}  mean|Δ|={mean:.3e}")

    print("--------------------------------------------")
    verdict = "PASS" if worst <= tol else "FAIL"
    print(f"worst max|Δ| = {worst:.3e}  (tol={tol:.1e})  ->  {verdict}")
    return worst <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip", "lxmert", "albef"])
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--checkpoint", default="flickr30k", help="ALBEF checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=None,
                    help="max allowed abs diff (defaults to a per-model value)")
    ap.add_argument("--env", choices=["both", "before", "after"], default="both")
    args = ap.parse_args()
    tol = args.tol if args.tol is not None else MODEL_TOL[args.model]

    stacks = ["before", "after"] if args.env == "both" else [args.env]
    results = {
        stack: run_stack(stack, args.model, args.num, args.checkpoint, args.seed)
        for stack in stacks
    }

    if args.env != "both":
        return
    ok = compare(results["before"], results["after"], tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
