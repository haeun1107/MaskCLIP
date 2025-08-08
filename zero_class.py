from scipy.sparse import load_npz

path = 'data/BTCV/label/ABD_001_61.npz'
arr = load_npz(path).toarray()

print("Loaded shape:", arr.shape)  # 👉 (13, 512*512) or (13, 512, 512)

if arr.shape == (13, 512 * 512):
    arr = arr.reshape(13, 512, 512)

print("Final label shape:", arr.shape)
