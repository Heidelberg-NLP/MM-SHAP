# conda activate shap (rampage)
import shap
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os, copy, json
import re, math, sys
import random
from tqdm import tqdm
from functools import partial

from ALBEF.models.vit import VisionTransformer
from ALBEF.models.xbert import BertConfig, BertModel
from ALBEF.models.tokenization_bert import BertTokenizer

from read_datasets import read_data


num_samples = sys.argv[1] # "all" or number
if num_samples != "all":
    num_samples = int(num_samples)
checkp = sys.argv[2] #  refcoco, mscoco, vqa, flickr30k
write_res = sys.argv[3] # "yes" or "no"
task = "image_sentence_alignment"  # image_sentence_alignment, vqa, gqa
other_tasks_than_valse = ['mscoco', 'vqa', 'gqa', 'gqa_balanced', 'nlvr2']
use_cuda = True

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
    # "gqa": ["/scratch/GQA/images/", "/scratch/GQA/test_all_questions.json"],
    # "nlvr2": ["/scratch/NLVR2/images", "/scratch/NLVR2/nlvr/nlvr2/data/test1.json"]
}


class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


def pre_caption(caption, max_words=30):
    """Text preprocessing for ALBEF."""
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])


def custom_masker(mask, x):
    """
    Shap relevant function.
    It gets a mask from the shap library with truth values about which image and text tokens to mask (False) and which not (True).
    It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the image, but have only defined which image tokens to mask. The image tokens masking happens in get_model_prediction().
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them)
    masked_X[0, 0] = 101  # start token ALBEF
    # masked_X[0, nb_text_tokens-1] = 4624 # sep token ALBEF (no TOKEN!!!)
    return masked_X


def get_model_prediction(x):
    """
    Shap relevant function.
    1. Mask the image pixel according to the specified patches to mask from the custom masker.
    2. Predict the model output for all combinations of masked image and tokens. This is then further passed to the shap libary.
    """
    with torch.no_grad():
        # split up the input_ids and the image_token_ids from x (containing both appended)
        input_ids = torch.tensor(x[:, :text_input.input_ids.shape[1]])
        masked_image_token_ids = torch.tensor(
            x[:, text_input.input_ids.shape[1]:])

        if use_cuda:
            input_ids = input_ids.cuda()
            masked_image_token_ids = masked_image_token_ids.cuda()

        # select / mask features and normalized boxes from masked_image_token_ids
        result = np.zeros(input_ids.shape[0])
        row_cols = 384 // patch_size 

        # call the model for each "new image" generated with masked features
        for i in range(input_ids.shape[0]):
            # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
            masked_text_inputs = text_input.copy()
            masked_text_inputs['input_ids'] = input_ids[i].unsqueeze(0)
            masked_image = copy.deepcopy(image)

            # pathify the image
            # torch.Size([1, 3, 384, 384]) image size ALBEF
            for k in range(masked_image_token_ids[i].shape[0]):
                if masked_image_token_ids[i][k] == 0:  # should be zero
                    m = k // row_cols  # 384 (img shape) / 16 (patch size)
                    n = k % row_cols
                    masked_image[:, :, m * patch_size:(m+1)*patch_size, n*patch_size:(
                        n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size) # np.random.rand()
            if use_cuda:
                outputs = model(masked_image.cuda(),
                                masked_text_inputs.to("cuda"))
            else:
                outputs = model(masked_image, masked_text_inputs)
            m = torch.nn.Softmax(dim=1)
            # this is the image-text similarity score
            result[i] = m(outputs).cpu().detach()[:, 1]
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
    model_path = f'ALBEF/checkpoints/{checkp}.pth'  # largest model: ALBEF.pth, smaller: ALBEF_4M.pth, refcoco, mscoco, vqa, flickr30k
    bert_config_path = 'ALBEF/configs/config_bert.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = VL_Transformer_ITM(
        text_encoder='bert-base-uncased', config_bert=bert_config_path)

    checkpoint = torch.load(model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    model.eval()

    block_num = 8

    model.text_encoder.base_model.base_model.encoder.layer[
        block_num].crossattention.self.save_attention = True

    if use_cuda:
        model.cuda()
    return model, tokenizer


model, tokenizer = load_models()

for instrument, foil_info in DATA.items():
    results = {'mmscore': {"captions": [], "foils": []},
               'accuracy': {"captions": [], "foils": []},
               'acc_r': []}
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
            elif instrument == 'mscoco':
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [foil["caption"], confounder["caption"]]
            else:
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [f'{foil["caption"]} {foil["answer"]}.', f'{confounder["caption"]} {confounder["answer"]}.']
            
            image_pil = Image.open(test_img_path).convert('RGB')
            image = transform(image_pil).unsqueeze(0)

            # shap values need one sentence for transformer
            for k, sentence in enumerate(test_sentences):

                # sentence = 'the woman is working on her computer at the desk'
                text = pre_caption(sentence)
                text_input = tokenizer(text, return_tensors="pt")

                if use_cuda:  # not yet if we want to paralelize shap
                    image = image.cuda()
                    text_input = text_input.to(image.device)

                model_prediction = model(image, text_input)
                m = torch.nn.Softmax(dim=1)
                img_sent_align_score = m(model_prediction).cpu().detach()[:, 1].item()

                if use_cuda:  # push back to cpu
                    image = image.cpu()
                    text_input = text_input.to(image.device)

                nb_text_tokens = text_input.input_ids.shape[1] # number of text tokens
                # calculate the number of patches needed to cover the image
                p = int(math.ceil(np.sqrt(nb_text_tokens)))
                patch_size = 384 // p # 384 is the image size for ALBEF
                image_token_ids = torch.tensor(range(1, p**2+1)).unsqueeze(0) # take one less because CLS and SEP tokens do not count

                # make a cobination between tokens and pixel_values (transform to patches first)
                X = torch.cat(
                    (text_input.input_ids, image_token_ids), 1).unsqueeze(1)

                # create an explainer with model and image masker
                explainer = shap.Explainer(
                    get_model_prediction, custom_masker, silent=True)
                shap_values = explainer(X)
                mm_score = compute_mm_score(nb_text_tokens, shap_values)

                if k == 0:
                    which = 'caption'
                    if img_sent_align_score >= 0.5:
                        results["accuracy"]["captions"].append(1)
                    else:
                        results["accuracy"]["captions"].append(0)
                    results["mmscore"]["captions"].append(mm_score)
                    img_sent_align_score_caption = img_sent_align_score
                else:
                    which = 'foil'
                    if img_sent_align_score < 0.5:
                        results["accuracy"]["foils"].append(1)
                    else:
                        results["accuracy"]["foils"].append(0)
                    results["mmscore"]["foils"].append(mm_score)
                    img_sent_align_score_foil = img_sent_align_score
                foil[f'{which}_albef_model_prediction'] = img_sent_align_score
                foil[f'{which}_albef_t_shap'] = mm_score

            if img_sent_align_score_caption > img_sent_align_score_foil:
                results["acc_r"].append(1)
            else:
                results["acc_r"].append(0)


    for what, mm_scores in results["mmscore"].items():
        if len(mm_scores) > 0:
            print(
                f"""We tested ALBEF {checkp} on {len(mm_scores)} samples of {instrument} {what}.
    The MM_score is: {np.array(mm_scores).mean()*100:.2f}% +/- {np.array(mm_scores).std()*100:.2f}% textual, the rest visual.
    The accuracy is: {np.array(results["accuracy"][what]).mean()*100:.2f}%.""")
    print(f"""The pairwise_accuracy is: {np.array(results["acc_r"]).mean()*100:.2f}%.
------""")

    # writing results down to a json file for further analysis of results on VALSE
    if write_res == 'yes':
        path = f"result_jsons/albef_{checkp}_{num_samples}/"
        os.makedirs(path, exist_ok=True)
        with open(f'result_jsons/albef_{checkp}_{num_samples}/{instrument}.json', 'w') as f:
            json.dump(foils_data, f)
