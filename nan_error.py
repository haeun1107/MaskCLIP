# 1. val set이 로드된 dataset 인스턴스를 불러와서 클래스 존재 여부 확인
from tqdm import tqdm
import numpy as np

# BTCVDataset을 이미 dataset = ... 으로 선언했다고 가정
class_idx = 13  # adrenal_gland_left
count = 0

for i in tqdm(range(len(dataset))):
    gt = dataset.get_gt_seg_map_by_idx(i)  # (512, 512)
    if (gt == class_idx).sum() > 0:
        count += 1

print(f"[INFO] Found {count} samples with adrenal_gland_left")
