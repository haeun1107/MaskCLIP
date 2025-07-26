# configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_btcv.py
_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv_13.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_4k.py'
]

suppress_labels = list(range(13))  # BTCV의 클래스 수 (1~13)

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        text_categories=13,
        #ignore_index=255,
        text_embeddings_path='pretrain/btcv_combined_RN50_clip_text.pth',
        clip_unlabeled_cats=suppress_labels
    )
)

find_unused_parameters=True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (512, 512)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type='BTCVDataset',
        data_root='data/BTCV',
        img_dir='image/x',
        ann_dir='label',
        split='train.txt',
        pipeline=train_pipeline
    )
)
