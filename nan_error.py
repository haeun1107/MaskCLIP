# from mmcv import Config
# from mmseg.datasets import build_dataset
# import numpy as np
# from tqdm import tqdm

# # config 경로는 실험에서 사용한 zero-shot config
# cfg = Config.fromfile('configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv3plus_r101-d8_480x480_40k_btcv.py')
# val_dataset = build_dataset(cfg.data.val)

# # target_class = 11  # adrenal_gland_left

# target_class = list(range(14))

# for t in range(14):
#     count = 0
#     for i in tqdm(range(len(val_dataset))):
#         gt = val_dataset.get_gt_seg_map_by_idx(i)
#         if (gt == t).sum() > 0:
#             count += 1

#     print(f"✅ [결과] 클래스 {t} 포함된 이미지 수: {count}")
    
from mmcv import Config
from mmseg.datasets import build_dataset
import numpy as np
from tqdm import tqdm

cfg = Config.fromfile('configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv3plus_r101-d8_480x480_40k_btcv.py')
train_dataset = build_dataset(cfg.data.train)

target_class = 12  # adrenal_gland_left
count = 0

for i in tqdm(range(len(train_dataset))):
    gt = train_dataset.get_gt_seg_map_by_idx(i)
    if (gt == target_class).sum() > 0:
        count += 1

print(f"✅ [Train] 클래스 13 (adrenal_gland_left) 포함된 이미지 수: {count}")

