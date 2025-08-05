# configs/_base_/datasets/btcv10.py
# BTCV Dataset for CustomDataset with .png images and .npz segmentation labels

dataset_type = 'BTCVDataset'
data_root = 'data/BTCV/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNpzAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = [
    'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
    'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
    'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right',
    'adrenal_gland_left'
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image/x',
        ann_dir='label',
        split='train_10.txt',  # ✅ 여기만 train_10.txt로 바꿈
        img_suffix='.png',
        seg_map_suffix='.npz',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image/x',
        ann_dir='label',
        split='val.txt',
        img_suffix='.png',
        seg_map_suffix='.npz',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image/x',
        ann_dir='label',
        split='val.txt',
        img_suffix='.png',
        seg_map_suffix='.npz',
        pipeline=test_pipeline,
        classes=classes),
)
