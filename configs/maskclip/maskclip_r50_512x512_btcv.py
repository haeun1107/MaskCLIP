# configs/maskclip/maskclip_r50_512x512_btcv.py

_base_ = [
    '../_base_/models/maskclip_r50.py',
    '../_base_/datasets/btcv.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

dataset_type = 'BTCVDataset'
data_root = 'data/BTCV/'

classes = [
    'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
    'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
    'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right',
    'adrenal_gland_left'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

img_size = (512, 512)

model = dict(
     decode_head=dict(
        type='MaskClipHead',
        in_channels=2048,
        in_index=3,
        input_transform=None,
        channels=1024,
        dropout_ratio=0.1,
        num_classes=13,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        text_categories=13,
        text_channels=1024,
        text_embeddings_path='pretrain/btcv_gpt_RN50_clip_text.pth',
        visual_projs_path='pretrain/RN50_clip_weights.pth',
        loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole')
)
