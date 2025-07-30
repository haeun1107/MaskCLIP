# configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_btcv_self.py

_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/btcv.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'  # ✅ 40K iteration
]

# BTCV has 14 classes (0~13), all suppressed in annotation-free setting
suppress_labels = list(range(13))  # ✅ all labels are unseen (annotation-free)

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        text_categories=13,
        text_embeddings_path='pretrain/btcv_re_RN50_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        cls_bg=False,
        decode_module_cfg=dict(
            type='DepthwiseSeparableASPPHead',
            input_transform=None,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
        ),
        reset_counter=True,                     # iteration 기준 초기화
        start_clip_guided=(1, 4000),            # 1 ~ 4000 : CLIP-guided only
        start_self_train=(4001, -1),            # 4001 이후 : Self-Training
    )
)


find_unused_parameters = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (512, 512)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=False),
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
