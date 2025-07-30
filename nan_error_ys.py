import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import os
from PIL import Image

LABEL_DIR = 'data/BTCV/label'
PRED_DIR = 'output/data/BTCV/image/x'  # 예측 PNG 위치
SAVE_DIR = 'visual'  # 저장할 디렉토리
os.makedirs(SAVE_DIR, exist_ok=True)

target_class = 13  # adrenal_gland_left
fname = 'ABD_033_75.npz'
base = fname.replace('.npz', '')

# Ground truth
gt_sparse = load_npz(os.path.join(LABEL_DIR, fname))
gt_dense = gt_sparse.toarray()
if gt_dense.shape[0] == 13:
    gt_dense = np.vstack([np.zeros((1, *gt_dense.shape[1:])), gt_dense])
gt_dense = gt_dense.reshape(14, 512, 512)
gt_seg = np.argmax(gt_dense, axis=0)

# Prediction
pred_path = os.path.join(PRED_DIR, base + '.png')
pred_seg = Image.open(pred_path).convert('L')  # Grayscale
pred_seg = np.array(pred_seg)

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gt_seg == target_class, cmap='Reds')
plt.title("GT")

plt.subplot(1, 3, 2)
plt.imshow(pred_seg == target_class, cmap='Blues')
plt.title("Prediction")

plt.subplot(1, 3, 3)
plt.imshow((gt_seg == target_class) | (pred_seg == target_class), cmap='Purples')
plt.title("Overlay")

plt.tight_layout()

# 파일로 저장
save_path = os.path.join(SAVE_DIR, f'vis_{base}.png')
plt.savefig(save_path)
print(f"✅ Visualization saved to: {save_path}")
