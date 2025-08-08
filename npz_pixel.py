import os
import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz

# ===== 경로 =====
LABEL_DIR = 'data/BTCV/label'
TRAIN_ALL_TXT = 'data/BTCV/train.txt'
TRAIN_10_TXT = 'data/BTCV/train_10.txt'

# ===== 클래스 이름 (foreground 13개) =====
CLASS_NAMES = {
    0: 'spleen',
    1: 'kidney_right',
    2: 'kidney_left',
    3: 'gallbladder',
    4: 'esophagus',
    5: 'liver',
    6: 'stomach',
    7: 'aorta',
    8: 'inferior_vena_cava',
    9: 'portal and splenic',
    10: 'pancreas',
    11: 'adrenal_gland_right',
    12: 'adrenal_gland_left'
}

# ===== 유틸 =====
def load_list(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def count_pixels_with_background(image_list, label_dir):
    class_counts = {i: 0 for i in range(13)}
    background_count = 0
    total_pixels = 0

    for img_name in tqdm(image_list, desc=f"Counting pixels in {len(image_list)} files"):
        npz_path = os.path.join(label_dir, img_name + '.npz')
        sparse = load_npz(npz_path)
        dense = sparse.toarray()

        if dense.shape == (13, 512 * 512):
            dense = dense.reshape(13, 512, 512)

        assert dense.shape == (13, 512, 512), f"Unexpected shape in {img_name}: {dense.shape}"

        # Foreground 클래스별 픽셀 수
        for cls_idx in range(13):
            count = np.count_nonzero(dense[cls_idx])
            class_counts[cls_idx] += count
            total_pixels += count

        # Background 픽셀 수 (모든 채널이 0인 픽셀)
        fg_mask = np.any(dense > 0, axis=0)
        bg_count = np.count_nonzero(~fg_mask)
        background_count += bg_count
        total_pixels += bg_count

    return class_counts, background_count, total_pixels

def print_counts_with_background(title, counts, background_count, total_pixels):
    print(f"\n[Pixel Counts] {title} (Total: {total_pixels:,} pixels)")
    for cls_idx in sorted(counts):
        name = CLASS_NAMES.get(cls_idx, 'Unknown')
        count = counts[cls_idx]
        percent = (count / total_pixels * 100) if total_pixels > 0 else 0.0
        print(f"Class {cls_idx:2d} ({name:20}): {count:10d} pixels ({percent:5.2f}%)")

    # Background 출력
    bg_percent = (background_count / total_pixels * 100) if total_pixels > 0 else 0.0
    print(f"Background             : {background_count:10d} pixels ({bg_percent:5.2f}%)")

# ===== 실행 =====
if __name__ == "__main__":
    all_list = load_list(TRAIN_ALL_TXT)
    ten_list = load_list(TRAIN_10_TXT)

    counts_all, bg_all, total_all = count_pixels_with_background(all_list, LABEL_DIR)
    counts_10, bg_10, total_10 = count_pixels_with_background(ten_list, LABEL_DIR)

    print_counts_with_background("Train All", counts_all, bg_all, total_all)
    print_counts_with_background("Train 10%", counts_10, bg_10, total_10)
