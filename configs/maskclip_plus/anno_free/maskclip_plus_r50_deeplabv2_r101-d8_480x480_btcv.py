# configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_btcv.py

_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv10.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_320k.py'
]

# === Label Suppression ===
# Treat all labels (including background and all 13 organs) as "unlabeled"
# This enables annotation-free learning using CLIP pseudo-labels
suppress_labels = list(range(13))  # BTCV 13 class

# === Model Configuration ===
model = dict(
    pretrained='open-mmlab://resnet101_v1c', # Pretrained weights for ResNet-101 backbone
    backbone=dict(depth=101),  # Use deeper ResNet-101 as backbone (instead of default ResNet-50)
    decode_head=dict(
        text_categories=13, # Number of text categories in CLIP: 13
        text_embeddings_path='pretrain/btcv_gpt_RN50_clip_text.pth', # Path to CLIP-generated text embeddings (should match 14 classes)
        clip_unlabeled_cats=suppress_labels,   # List of categories where no annotation is available; use CLIP instead
        # No self-training phase here (start_self_train is omitted)
        cls_bg=False,  # Don't Use background text embedding (e.g., for label 0)
        decode_module_cfg=dict(
            type='DepthwiseSeparableASPPHead',  # memory-efficient ASPP head
            input_transform=None,
            dilations=(1, 12, 24, 36), # dilation rates for ASPP
            c1_in_channels=256, # input channel for low-level features
            c1_channels=48, # channel reduction for skip connection
        ),
    )
)

# multi-GPU 학습 시 파라미터 사용 유무를 명시
find_unused_parameters = True

# Normalize 설정
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 입력 크기 및 전처리 파이프라인
img_scale = (512, 512)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=False, suppress_labels=suppress_labels),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
