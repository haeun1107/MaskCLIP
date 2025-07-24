import torch
import clip
import argparse
import os

# Natural language descriptions (approx. 60 tokens each)
btcv_prompts = {
    "background": "everything in the image that is not part of a known organ or anatomical structure, including fat, muscle, and surrounding tissues.",
    "spleen": "the spleen: a soft, fist-sized organ that filters blood and helps fight infections in the immune system.",
    "kidney_right": "the right kidney: a bean-shaped organ located in the right side of the abdomen, responsible for filtering waste from blood and producing urine.",
    "kidney_left": "the left kidney: a bean-shaped organ located in the left side of the abdomen that maintains fluid balance and removes toxins from the body.",
    "gallbladder": "the gallbladder: a small, pear-shaped organ under the liver that stores bile to aid in fat digestion.",
    "esophagus": "the esophagus: a muscular tube connecting the throat to the stomach, allowing food and liquids to pass through.",
    "liver": "the liver: a large reddish-brown organ that detoxifies chemicals, metabolizes drugs, and produces bile for digestion.",
    "stomach": "the stomach: a muscular, hollow organ that breaks down food using digestive acids and enzymes.",
    "aorta": "the aorta: the largest artery in the body that carries oxygen-rich blood from the heart to the rest of the body.",
    "inferior_vena_cava": "the inferior vena cava: a large vein that carries deoxygenated blood from the lower body back to the heart.",
    "portal_vein_and_splenic_vein": "the portal and splenic veins: blood vessels that transport nutrient-rich blood from the gastrointestinal tract and spleen to the liver.",
    "pancreas": "the pancreas: an elongated gland behind the stomach that helps with digestion and regulates blood sugar levels.",
    "adrenal_gland_right": "the right adrenal gland: a small gland sitting above the right kidney that produces hormones like adrenaline and cortisol.",
    "adrenal_gland_left": "the left adrenal gland: a hormone-secreting gland located above the left kidney, involved in stress response and metabolism."
}

def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engineering for BTCV zero-shot segmentation')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='CLIP model name')
    parser.add_argument('--output', default='pretrain/btcv_gpt_RN50_clip_text.pth', help='Output path to save the CLIP text embeddings')
    return parser.parse_args()

def encode_prompts(prompts_dict, model_name):
    model, _ = clip.load(model_name)
    class_names = list(prompts_dict.keys())
    descriptions = list(prompts_dict.values())

    with torch.no_grad():
        texts = clip.tokenize(descriptions).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    return class_embeddings, class_names

if __name__ == '__main__':
    args = parse_args()
    model_map = {
        'RN50': 'RN50',
        'RN101': 'RN101',
        'RN50x4': 'RN50x4',
        'RN50x16': 'RN50x16',
        'ViT32': 'ViT-B/32',
        'ViT16': 'ViT-B/16'
    }

    embeddings, class_names = encode_prompts(btcv_prompts, model_map[args.model])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(embeddings.float(), args.output)
    print(f"Saved {len(class_names)} class embeddings to {args.output}")
