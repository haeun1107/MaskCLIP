import os
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

LABEL_DIR = 'data/BTCV/label'
target_class = 13  # class index for adrenal_gland_left (after adding background)
files_with_target_class = []

print(f"🔍 Checking for class {target_class} (adrenal_gland_left) based on dense sum...\n")

for fname in tqdm(os.listdir(LABEL_DIR)):
    if not fname.endswith('.npz'):
        continue

    npz_path = os.path.join(LABEL_DIR, fname)
    try:
        sparse_arr = load_npz(npz_path)
        dense_arr = sparse_arr.toarray()  # shape: (13, H*W) or (13, H, W)

        # reshape if needed
        if dense_arr.shape == (13, 512 * 512):
            dense_arr = dense_arr.reshape(13, 512, 512)

        if dense_arr.shape[0] == 13:
            # Add background channel at index 0 → push class 12 → 13
            dense_arr = np.vstack([np.zeros((1, *dense_arr.shape[1:])), dense_arr])

        if dense_arr.shape[0] != 14:
            print(f"❌ Shape mismatch in {fname}: {dense_arr.shape}")
            continue

        # Check if class 13 (adrenal_gland_left) appears
        if dense_arr[target_class].sum() > 0:
            files_with_target_class.append(fname)

    except Exception as e:
        print(f"⚠️ Error reading {fname}: {e}")

# 🔽 결과 출력
print(f"\n📊 Found {len(files_with_target_class)} files containing class {target_class} (adrenal_gland_left):")
for f in files_with_target_class:
    print(" -", f)

if not files_with_target_class:
    print("\n❌ Class 13 never appears in the current dataset!")
else:
    print("\n✅ Class 13 is present in some label files.")
