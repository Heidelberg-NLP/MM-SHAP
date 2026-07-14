"""Evaluate a model with MM-SHAP over the VALSE foil benchmark.

For each example VALSE pairs a correct caption with a minimally different foil. We run
MM-SHAP on both, record each sentence's matching score and MM-score, and report the
mean MM-score plus pairwise accuracy (does the model score the caption above the foil?).
"""
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmshap.core import MMShapModel, mm_shap_score
from read_datasets import read_data

MTURK_MIN_VOTES = 2  # keep only examples whose caption enough annotators accepted
ROLES = ("caption", "foil")

# instrument -> [images dir, annotations json]; add more VALSE instruments as needed.
VALSE_DATA = {"existence": ["data/foil-benchmark/images/existence",
                            "data/foil-benchmark/annotations/existence.sample.json"]}


def select_sentences(instrument: str, foil: dict) -> list[str]:
    """The correct caption and its foil for a VALSE example."""
    caption = foil["caption"][0] if instrument == "plurals" else foil["caption"]
    return [caption, foil["foils"][0]]


def evaluate(model: MMShapModel, instrument: str, images_path: str,
             annotations_path: str, num_samples) -> dict:
    data = read_data(instrument, annotations_path, images_path)
    if num_samples != "all":
        data = dict(random.sample(list(data.items()), num_samples))

    mm_scores = {role: [] for role in ROLES}
    accuracy = {role: [] for role in ROLES}
    pairwise = []

    for foil in tqdm(list(data.values())):
        if foil["mturk"]["caption"] < MTURK_MIN_VOTES:
            continue
        image_ctx = model.encode_image(str(Path(images_path) / foil["image_file"]))
        predictions = _score_pair(model, image_ctx, instrument, foil,
                                  mm_scores, accuracy)
        if len(predictions) == len(ROLES):
            pairwise.append(int(predictions["caption"] > predictions["foil"]))

    return {"mm_scores": mm_scores, "accuracy": accuracy,
            "pairwise": pairwise, "data": data}


def _score_pair(model: MMShapModel, image_ctx, instrument: str, foil: dict,
                mm_scores: dict, accuracy: dict) -> dict:
    predictions = {}
    for role, sentence in zip(ROLES, select_sentences(instrument, foil)):
        sample = model.preprocess(image_ctx, sentence)
        if sample is None:
            continue
        prediction = model.predict(sample)
        score = mm_shap_score(model, sample)

        mm_scores[role].append(score)
        predictions[role] = prediction
        if model.reports_accuracy:
            hit = prediction >= 0.5 if role == "caption" else prediction < 0.5
            accuracy[role].append(int(hit))
        foil[f"{role}_{model.key}_model_prediction"] = prediction
        foil[f"{role}_{model.key}_t_shap"] = score
    return predictions


def report(model: MMShapModel, instrument: str, results: dict) -> None:
    for role in ROLES:
        scores = results["mm_scores"][role]
        if not scores:
            continue
        line = (f"We tested {model.title} on {len(scores)} samples of "
                f"{instrument} {role}s.\n"
                f"    The MM_score is: {np.mean(scores) * 100:.2f}% "
                f"+/- {np.std(scores) * 100:.2f}% textual, the rest visual.")
        if model.reports_accuracy:
            acc = np.mean(results["accuracy"][role]) * 100
            line += f"\n    The accuracy is: {acc:.2f}%."
        print(line)
    pairwise_acc = np.mean(results["pairwise"]) * 100
    print(f"The pairwise_accuracy is: {pairwise_acc:.2f}%.\n------")


def write_results(model: MMShapModel, instrument: str, run_id: str, data: dict) -> None:
    out_dir = Path("result_jsons") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{instrument}.json").write_text(json.dumps(data))


def run(model: MMShapModel, data_config: dict, num_samples, write_res: bool,
        run_id_suffix: str = "") -> None:
    run_id = f"{model.key}{run_id_suffix}_{num_samples}"
    for instrument, (images_path, annotations_path) in data_config.items():
        results = evaluate(model, instrument, images_path,
                           annotations_path, num_samples)
        report(model, instrument, results)
        if write_res:
            write_results(model, instrument, run_id, results["data"])
