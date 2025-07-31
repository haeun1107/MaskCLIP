import numpy as np
import torch
import clip
import argparse

bg_classes = ['building', 'ground', 'grass', 'tree', 'sky']

btcv_classes = ['spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left']

prompt_templates = [
    "a photo of the {}.",
    "a close-up photo of the {}.",
    "a medical image of a {}.",
    "an MRI scan of the {}.",
    "a CT image showing the {}.",
    "a detailed cross-sectional image of the {}.",
    "a radiology image containing a {}.",
    "a grayscale image of the {}.",
    "a diagnostic image of a {}.",
    "a low-contrast scan of the {}.",

    "the {} seen in a CT scan.",
    "a 3D rendering of the {} in a CT volume.",
    "a scan of a patient's {}.",
    "the {} organ inside the human body.",
    "an internal view of the {}.",
    "a segmented image highlighting the {}.",
    "a blurry CT scan showing the {}.",
    "an organ scan focusing on the {}.",
    "a scan of abdominal region with {}.",
    "a medical slice showing the {} clearly.",

    "the {} visible in abdominal imaging.",
    "a slice-level view of the {}.",
    "cross-sectional visualization of the {}.",
    "the anatomical region containing the {}.",
    "a clinical image focusing on the {}.",
    "axial plane scan of the {}.",
    "a DICOM image including the {}.",
    "a coronal view showing the {} organ.",
    "segmentation output for the {}.",
    "the {} depicted in a contrast-enhanced CT.",

    "there is a {} in the body scan.",
    "this scan includes the {}.",
    "this is the {} in the image.",
    "this medical image contains the {} region.",
    "this image highlights the patient's {}.",
    "you can see the {} here.",
    "focus of this image is the {}.",
    "this scan clearly shows the {}.",
    "region of interest: the {}.",
    "highlighted organ: the {}.",

    "a CT scan showing the {}.",
    "a medical image of the {}.",
    "an MRI scan of the {}.",
    "a scan highlighting the {}.",
    "a diagnostic image showing the {}.",
    "a DICOM image containing the {}.",
    "a radiology scan of the {}.",
    "a cross-sectional scan of the {}.",
    "a scan of the {} region.",
    "an abdominal scan with the {}.",
    "an axial slice of the {}.",
    "a coronal slice showing the {}.",
    "the {} in a contrast-enhanced scan.",
    "a scan containing the {} organ.",
    "the {} depicted in CT imaging.",
    "a segmentation map for the {}.",
    "organ segmentation of the {}.",
    "a 3D volume scan showing the {}.",
    "a body slice focusing on the {}.",
    "CT view of the {}.",

    "an organ called the {}.",
    "an internal view of the {}.",
    "the anatomical structure: {}.",
    "inside the body, there is the {}.",
    "the {} is surrounded by other organs.",
    "a 2D slice of the {} inside the body.",
    "an example of the {} in human anatomy.",
    "the {} connects to major vessels.",
    "the {} is near the spine in this image.",
    "the {} is shown near the liver.",
    "a close-up of the {} organ.",
    "the {} can be identified in this scan.",
    "a view of the {} with soft tissue contrast.",
    "the {} shown in anatomical position.",
    "imaging slice showing the {} organ.",
    "organ detail: the {}.",
    "a CT volume section with the {}.",
    "radiological view of the {}.",
    "contrast-filled image of the {}.",
    "human body slice featuring the {}.",

]

def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engeering script')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='clip model name')
    parser.add_argument('--class-set', default=['voc'], nargs='+',
        choices=['btcv'],
        help='the set of class names')
    parser.add_argument('--no-prompt-eng', action='store_true', help='disable prompt engineering')

    args = parser.parse_args()
    return args

def zeroshot_classifier(model_name, classnames, templates):
    model, preprocess = clip.load(model_name)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

if __name__ == '__main__':
    args = parse_args()

    classes = []
    all_set_name = ''
    name_mapping = {'btcv': btcv_classes}
    for set_name in args.class_set:
        if set_name in name_mapping:
            classes += name_mapping[set_name]
            all_set_name += '_{}'.format(set_name)
        if set_name in ['blur'] or args.no_prompt_eng:
            prompt_templates = ['a photo of a {}.']
    # remove redundant classes
    classes = list(dict.fromkeys(classes))
    # remove the first underline
    all_set_name = all_set_name[1:]
    print(classes)

    print(f"{len(classes)} class(es), {len(prompt_templates)} template(s)")

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16'}
    zeroshot_weights = zeroshot_classifier(name_mapping[args.model], classes, prompt_templates)
    zeroshot_weights = zeroshot_weights.permute(1, 0).float()
    print(zeroshot_weights.shape)

    prefix = f'pretrain/{all_set_name}_{args.model}'
    if args.no_prompt_eng:
        prefix += '_npe'

    torch.save(zeroshot_weights, f'{prefix}_vr_clip_text.pth')

