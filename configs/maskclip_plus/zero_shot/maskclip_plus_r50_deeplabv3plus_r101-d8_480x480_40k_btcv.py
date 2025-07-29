_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'
]

suppress_labels = [1, 4, 7]  # spleen, gallbladder, stomach

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        text_categories=14,  # ✅ background 포함
        #text_embeddings_path='pretrain/btcv_gpt_RN50_clip_text.pth',
        #text_embeddings_path='pretrain/btcv_RN50_clip_text.pth',
        #text_embeddings_path='pretrain/btcv_clip_RN50_clip_text.pth',
        #text_embeddings_path='pretrain/btcv_combined_RN50_clip_text.pth',
        text_embeddings_path='pretrain/btcv_RN50_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        unlabeled_cats=suppress_labels,
        start_clip_guided=(1, 3999),
        start_self_train=(4000, -1),
        cls_bg=True,  # ✅ background 포함이면 꼭 True
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (512, 512)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', suppress_labels=suppress_labels, reduce_zero_label=False),
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
