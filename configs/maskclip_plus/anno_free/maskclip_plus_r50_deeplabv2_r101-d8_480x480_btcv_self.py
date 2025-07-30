_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'  # ✅ 40K iteration
]

# BTCV has 14 classes (0~13), all suppressed in annotation-free setting
suppress_labels = list(range(0, 14))  # ✅ all labels are unseen (annotation-free)

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        text_categories=14,
        text_embeddings_path='pretrain/btcv_RN50_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,  # ✅ treat all classes as unseen
        unlabeled_cats=suppress_labels,       # ✅ enable self-training with pseudo labels
        start_clip_guided=(1, 3999),          # ✅ phase 1: clip-guided only
        start_self_train=(4000, -1),          # ✅ phase 2: self-training until 40K
        cls_bg=True,                          # ✅ include background in label index
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
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(480, 480), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(480, 480), pad_val=0, seg_pad_val=255),
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
