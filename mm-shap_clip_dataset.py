# conda activate shap (rampage)
import shap
import torch
import numpy as np
from PIL import Image
import os, copy, sys
import math, json
import random
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

from read_datasets import read_data

# sys.stderr = open('results_txts/clip_all_samples.txt', 'a')

num_samples = sys.argv[1] # "all" or number
if num_samples != "all":
    num_samples = int(num_samples)
write_res = sys.argv[2] # "yes" or "no"
task = "image_sentence_alignment"  # image_sentence_alignment, vqa, gqa
other_tasks_than_valse = ['mscoco', 'vqa', 'gqa', 'gqa_balanced', 'nlvr2']

DATA = {
    # "foil_it": ["/scratch/COCO/val2014/",
    #             "/scratch/foil-benchmark/orig_foil/foil_it_test_mturk.json"],
    "existence": ["/scratch/visualglue-data-collection/visual7w/images/",
                  '/scratch/foil-benchmark/existence/existence_benchmark.test_mturk.json'],
    # "plurals": ["/scratch/foil-benchmark/plurals/test_images/",
    #             '/scratch/foil-benchmark/plurals/plurals_test_mturk.json'],
    # "counting_hard": ["/scratch/visualglue-data-collection/visual7w/images/",
    #                   '/scratch/foil-benchmark/counting_hard/visual7w_counting.hard.test_mturk.json'],
    # "counting_small": ['/scratch/visualglue-data-collection/visual7w/images/',
    #                    '/scratch/foil-benchmark/counting/visual7w_counting.small-quantities.test_mturk.json'],
    # "counting_adversarial": ["/scratch/visualglue-data-collection/visual7w/images/",
    #                          '/scratch/foil-benchmark/counting_adversarial/visual7w_counting.adversarial.test_mturk.json'],
    # "relations": ["/scratch/foil-benchmark/relations/test_images/",
    #               '/scratch/foil-benchmark/relations/relations_test_mturk.json'],
    # "action replace": ['/scratch/foil-benchmark/actions/images_512/',
    #                    '/scratch/foil-benchmark/actions/action_replace/action_replace_test_mturk.json'],
    # "actant swap": ['/scratch/foil-benchmark/actions/images_512/',
    #                 '/scratch/foil-benchmark/actions/actant_swap/actant_swap_test_mturk.json'],
    # "coref": ["/scratch/foil-benchmark/coref/release_too_many_is_this_in_color/images/",
    #           '/scratch/foil-benchmark/coref/coref_test_visdial_train_mturk.json'],
    # "coref_hard": ["/scratch/foil-benchmark/coref/release_v18/test_images/",
    #                '/scratch/foil-benchmark/coref/coref_test_hard_mturk.json'],
    # "mscoco": ["/scratch/COCO/val2014/", "/scratch/foil-benchmark/orig_foil/foil_it_test_mturk.json"],
    # "vqa": ["/scratch/COCO/val2014/", "/scratch/VQA2.0/v2_OpenEnded_mscoco_val2014_questions.json"],
    # "gqa": ["/scratch/GQA/images/", "/scratch/GQA/val_all_questions.json"],
    # "gqa_balanced": ["/scratch/GQA/images/", "/scratch/GQA/val_balanced_questions.json"],
    # "nlvr2": ["/scratch/NLVR2/images", "/scratch/NLVR2/nlvr/nlvr2/data/test1.json"]
}


def custom_masker(mask, x):
    """
    Shap relevant function. Defines the masking function so the shap computation
    can 'know' how the model prediction looks like when some tokens are masked.
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them)
    masked_X[0, 0] = 49406
    masked_X[0, text_length_tok-1] = 49407
    return masked_X


def get_model_prediction(x):
    """
    Shap relevant function. Predict the model output for all combinations of masked tokens.
    """
    with torch.no_grad():
        # split up the input_ids and the image_token_ids from x (containing both appended)
        input_ids = torch.tensor(x[:, :inputs.input_ids.shape[1]])
        masked_image_token_ids = torch.tensor(x[:, inputs.input_ids.shape[1]:])

        # select / mask features and normalized boxes from masked_image_token_ids
        result = np.zeros(input_ids.shape[0])

        row_cols = 224 // patch_size # 224 / 32 = 7

        # call the model for each "new image" generated with masked features
        for i in range(input_ids.shape[0]):
            # here the actual masking of CLIP is happening. The custom masker only specified which patches to mask, but no actual masking has happened
            masked_inputs = copy.deepcopy(inputs)  # initialize the thing
            masked_inputs['input_ids'] = input_ids[i].unsqueeze(0)

            # pathify the image
            # torch.Size([1, 3, 224, 224]) image size CLIP
            for k in range(masked_image_token_ids[i].shape[0]):
                if masked_image_token_ids[i][k] == 0:  # should be zero
                    m = k // row_cols
                    n = k % row_cols
                    masked_inputs["pixel_values"][:, :, m *
                        patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()
            
            outputs = model(**masked_inputs)
            # CLIP does not work with probabilities, because these are computed with softmax among choices (which I do not have here)
            # this is the image-text similarity score
            result[i] = outputs.logits_per_image
    return result


def compute_mm_score(text_length, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    text_contrib = np.abs(shap_values.values[0, 0, :text_length]).sum()
    image_contrib = np.abs(shap_values.values[0, 0, text_length:]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score


def load_models():
    """ Load models and model components. """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_models()

for instrument, foil_info in DATA.items():
    results = {'text_score': {"captions": [], "foils": []},  'acc_r': []}
    images_path = foil_info[0]
    foils_path = foil_info[1]

    foils_data = read_data(instrument, foils_path, images_path)
    # subsample the data (for faster estimates), to test code for a few samples    
    random.seed(1520)
    if num_samples != "all":
        foils_data = dict(random.sample(foils_data.items(), num_samples))  # 100

    for foil_id, foil in tqdm(foils_data.items()):  # tqdm
        if instrument not in other_tasks_than_valse:
            caption_fits = foil['mturk']['caption']
        else:
            # pretend like the sample was accepted by annotators (for everything other than VALSE)
            caption_fits = 3
        if caption_fits >= 2:  # MTURK filtering! Use only valid set

            test_img_path = os.path.join(images_path, foil["image_file"])

            # work with one sentence at a time to avoid attention mask and image features confusions.
            if instrument not in other_tasks_than_valse:
                if instrument == 'plurals':
                    test_sentences = [foil["caption"][0], foil["foils"][0]]
                else:
                    test_sentences = [foil["caption"], foil["foils"][0]]
                # take only captions !!!!FOILS
            elif instrument == 'mscoco':
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [foil["caption"], confounder["caption"]]
            else:
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [f'{foil["caption"]} {foil["answer"]}.', f'{confounder["caption"]} {confounder["answer"]}.']

            image = Image.open(test_img_path)

            # shap values need one sentence for transformer
            for k, sentence in enumerate(test_sentences):

                try:  # image feature extraction can go wrong
                    inputs = processor(
                        text=sentence, images=image, return_tensors="pt", padding=True
                    )
                except:
                    continue
                model_prediction = model(**inputs).logits_per_image[0,0].item()

                text_length_tok = inputs.input_ids.shape[1]
                p = int(math.ceil(np.sqrt(text_length_tok)))
                patch_size = 224 // p
                image_token_ids = torch.tensor(
                    range(1, p**2+1)).unsqueeze(0) # (inputs.pixel_values.shape[-1] // patch_size)**2 +1
                # make a cobination between tokens and pixel_values (transform to patches first)
                X = torch.cat(
                    (inputs.input_ids, image_token_ids), 1).unsqueeze(1)

                # create an explainer with model and image masker
                explainer = shap.Explainer(
                    get_model_prediction, custom_masker, silent=True)
                shap_values = explainer(X)
                mm_score = compute_mm_score(text_length_tok, shap_values)
                
                if k == 0:
                    which = 'caption'
                    results["text_score"]["captions"].append(mm_score)
                    model_prediction_caption = model_prediction
                else:
                    which = 'foil'
                    results["text_score"]["foils"].append(mm_score)
                    model_prediction_foil = model_prediction
                # clip can only work with pairwise accuracy
                foil[f'{which}_clip_model_prediction'] = model_prediction
                foil[f'{which}_clip_t_shap'] = mm_score
                
            if model_prediction_caption > model_prediction_foil:
                results["acc_r"].append(1)
            else:
                results["acc_r"].append(0)


    for what, mm_scores in results["text_score"].items():
        if len(mm_scores) > 0:
            print(
                f"""We tested CLIP on {len(mm_scores)} samples of {instrument} {what}.
    The MM_score is: {np.array(mm_scores).mean()*100:.2f}% +/- {np.array(mm_scores).std()*100:.2f}% textual, the rest visual.""")
    print(f"""The pairwise_accuracy is: {np.array(results["acc_r"]).mean()*100:.2f}%.
------""")

    # writing results down to a json file for further analysis of results on VALSE
    if write_res == 'yes':
        path = f"result_jsons/clip_{num_samples}/"
        os.makedirs(path, exist_ok=True)
        with open(f'result_jsons/clip_{num_samples}/{instrument}.json', 'w') as f:
            json.dump(foils_data, f)
