# configs/maskclip/maskclip_r50_512x512_btcv.py

_base_ = [
    '../_base_/models/maskclip_r50.py',
    '../_base_/datasets/btcv10.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py'
]

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
        text_embeddings_path='pretrain/btcv_gpt_RN50_clip_text.pth',  # 👉 text embedding 사용 안 함
        visual_projs_path='pretrain/RN50_clip_weights.pth',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    test_cfg=dict(mode='whole')
)