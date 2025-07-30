import torch
import clip
import argparse
import os

# 1. Natural language descriptions
btcv_prompts = {
    "background": "everything in the image that is not part of a known organ or anatomical structure, including fat, muscle, and surrounding tissues.",
    "spleen": "a soft, fist-sized organ that filters blood and helps fight infections in the immune system.",
    "kidney_right": "a bean-shaped organ located in the right side of the abdomen, responsible for filtering waste from blood and producing urine.",
    "kidney_left": "a bean-shaped organ located in the left side of the abdomen that maintains fluid balance and removes toxins from the body.",
    "gallbladder": "a small, pear-shaped organ under the liver that stores bile to aid in fat digestion.",
    "esophagus": "a muscular tube connecting the throat to the stomach, allowing food and liquids to pass through.",
    "liver": "a large reddish-brown organ that detoxifies chemicals, metabolizes drugs, and produces bile for digestion.",
    "stomach": "a muscular, hollow organ that breaks down food using digestive acids and enzymes.",
    "aorta": "the largest artery in the body that carries oxygen-rich blood from the heart to the rest of the body.",
    "inferior_vena_cava": "a large vein that carries deoxygenated blood from the lower body back to the heart.",
    "portal_vein_and_splenic_vein": "blood vessels that transport nutrient-rich blood from the gastrointestinal tract and spleen to the liver.",
    "pancreas": "an elongated gland behind the stomach that helps with digestion and regulates blood sugar levels.",
    "adrenal_gland_right": "a small gland sitting above the right kidney that produces hormones like adrenaline and cortisol.",
    "adrenal_gland_left": "a hormone-secreting gland located above the left kidney, involved in stress response and metabolism."
}

# 2. Renamed class names
btcv_renamed_classes = {
    "background": "non-organ area",
    "spleen": "blood-filtering immune organ",
    "kidney_right": "right renal organ",
    "kidney_left": "left renal organ",
    "gallbladder": "bile-storing sac under the liver",
    "esophagus": "food-passing muscular tube",
    "liver": "large detoxifying organ",
    "stomach": "acid-filled digestive organ",
    "aorta": "main oxygen-carrying artery",
    "inferior_vena_cava": "large blood-returning vein",
    "portal_vein_and_splenic_vein": "nutrient-rich portal vessels",
    "pancreas": "insulin-producing elongated gland",
    "adrenal_gland_right": "right hormone-producing gland",
    "adrenal_gland_left": "left hormone-producing gland"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Combine rewritten class names with natural descriptions for CLIP text embedding')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='CLIP model name')
    parser.add_argument('--output', default='pretrain/btcv_combined_bg_RN50_clip_text.pth', help='Output path for text embeddings')
    return parser.parse_args()

def build_combined_prompts(name_map, desc_map):
    prompts = []
    for key in name_map:
        name = name_map[key]
        desc = desc_map[key]
        combined = f"{name}: {desc}"
        prompts.append(combined)
    return prompts, list(name_map.keys())

def encode_text_prompts(text_list, model_name):
    model, _ = clip.load(model_name)
    with torch.no_grad():
        tokens = clip.tokenize(text_list).cuda()
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

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

    # ⬇️ Combine class name and description into one prompt string
    combined_prompts, class_keys = build_combined_prompts(btcv_renamed_classes, btcv_prompts)
    embeddings = encode_text_prompts(combined_prompts, model_map[args.model])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(embeddings.float(), args.output)

    print(f"✅ Saved {len(class_keys)} combined class embeddings to {args.output}")
