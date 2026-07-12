"""Command-line entry point for MM-SHAP.

Runs the MM-SHAP analysis for one model (CLIP, LXMERT, or ALBEF) over the VALSE foil
benchmark: for each (image, sentence) pair it reports how much of the model's
image-text matching score comes from the text vs. the image, plus pairwise
caption-vs-foil accuracy. The heavy lifting lives in the ``mmshap`` package; this file
only parses arguments, seeds for determinism, builds the requested model, and hands
off to ``mmshap.evaluation.run``.

Must be run from the repository root so that ``import shap`` resolves to the vendored
``shap/`` package (a modified copy) rather than any pip-installed shap:

    uv run python mm-shap.py clip 20 --write
    uv run python mm-shap.py lxmert all --batch-size 16
    uv run python mm-shap.py albef 20 --checkpoint flickr30k --write

``--batch-size`` sets how many masked (image, sentence) variants are scored per forward
pass. Larger values are faster but use more GPU memory; lower it if you hit OOM.

``--checkpoint`` (ALBEF only) selects which finetuned ALBEF weights to load, i.e. the
full image-text-matching model (ViT-B/16 visual encoder + BERT + ITM head) finetuned on
that dataset -- e.g. ``flickr30k`` or ``mscoco``. It is ignored for CLIP and LXMERT.
"""
import argparse

from mmshap.core import MMShapModel
from mmshap.evaluation import VALSE_DATA, run
from mmshap_repro import maybe_seed


def build_model(name: str, checkpoint: str) -> MMShapModel:
    if name == "clip":
        from mmshap.models.clip import Clip
        return Clip()
    if name == "lxmert":
        from mmshap.models.lxmert import Lxmert
        return Lxmert()
    from mmshap.models.albef import Albef
    return Albef(checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["clip", "lxmert", "albef"])
    ap.add_argument("num_samples", help='"all" or an integer')
    ap.add_argument("--checkpoint", default="flickr30k",
                    help="finetuned ALBEF weights to load (e.g. flickr30k, mscoco)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="masked variants scored per forward pass; raise to go faster, "
                         "lower if the GPU runs out of memory")
    ap.add_argument("--write", action="store_true", help="write result jsons")
    args = ap.parse_args()

    maybe_seed()  # deterministic only if MMSHAP_SEED is set (used by the harness)
    num = args.num_samples
    num_samples = num if num == "all" else int(num)
    suffix = f"_{args.checkpoint}" if args.model == "albef" else ""
    model = build_model(args.model, args.checkpoint)
    model.masked_batch_size = args.batch_size
    run(model, VALSE_DATA, num_samples, args.write, run_id_suffix=suffix)


if __name__ == "__main__":
    main()
