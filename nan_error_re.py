import os
import numpy as np
from scipy.sparse import load_npz

# ===== 라벨 경로 =====
LABEL_FILE = 'data/BTCV/label/ABD_004_97.npz'  # 파일 경로 수정

# ===== 로드 및 구조 확인 =====
sparse = load_npz(LABEL_FILE)
dense = sparse.toarray()

# (13, H*W) → (13, H, W) 형태로 변환
if dense.shape[0] == 13 and dense.shape[1] == 512 * 512:
    dense = dense.reshape(13, 512, 512)

print(f"라벨 shape: {dense.shape}")  # 예상: (13, 512, 512)

# ===== 채널 수 검증 =====
num_channels = dense.shape[0]
print(f"채널 수: {num_channels}")
assert num_channels == 13, "⚠️ 채널 수가 13이 아님!"

# ===== 픽셀별 클래스 ID 계산 =====
# argmax로 각 픽셀의 채널 index를 클래스 ID로 변환
class_map = np.argmax(dense, axis=0)  # shape: (512, 512)

# background 픽셀 찾기 (모든 채널이 0인 경우)
bg_mask = (dense.sum(axis=0) == 0)
class_map[bg_mask] = 255  # background를 255로 표시

# ===== 통계 출력 =====
unique, counts = np.unique(class_map, return_counts=True)
print("\n클래스 ID 분포 (255=background):")
for cls_id, cnt in zip(unique, counts):
    print(f"Class {cls_id:3d}: {cnt} pixels ({cnt / class_map.size * 100:.2f}%)")
