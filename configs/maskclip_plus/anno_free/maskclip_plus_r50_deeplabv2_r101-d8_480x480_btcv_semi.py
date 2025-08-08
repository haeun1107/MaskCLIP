# configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_btcv_semi.py

_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv_semi.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_100k.py'
]

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101), 
    decode_head=dict(
        text_categories=13, 
        text_embeddings_path='pretrain/btcv_gpt_RN50_clip_text.pth',
        cls_bg=False,
        decode_module_cfg=dict(
            type='DepthwiseSeparableASPPHead',
            input_transform=None,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256, 
            c1_channels=48, 
        ),
    )
)

find_unused_parameters = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

img_scale = (512, 512)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
