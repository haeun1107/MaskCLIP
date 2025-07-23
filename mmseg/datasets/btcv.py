# MaskCLIP/mmseg/datasets/btcv.py
import os.path as osp
import numpy as np
import json
from .builder import DATASETS
from .custom import CustomDataset
from scipy.sparse import load_npz

@DATASETS.register_module()
class BTCVDataset(CustomDataset):
    """BTCV Dataset with .npz label and CLIP feature .npy input."""

    CLASSES = [
        'background', 'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_vein_and_splenic_vein', 'pancreas',
        'adrenal_gland_right', 'adrenal_gland_left'
    ]

    PALETTE = [[i * 20, i * 20, i * 20] for i in range(14)]

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)  # ✅ split 직접 넘기기


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None, **kwargs):
        with open(self.split, 'r') as f:
            lines = f.readlines()

        data_infos = []
        for line in lines:
            base = line.strip()
            data_infos.append(dict(
                img_info=dict(
                    filename=osp.join(img_dir, base + img_suffix),
                    img_prefix=None
                ),
                ann_info=dict(
                    seg_map=osp.join(ann_dir, base + seg_map_suffix),
                    seg_prefix=None
                )
            ))
        #print("[DEBUG] First sample in data_infos:", data_infos[0])
        print(f"[INFO] Loaded {len(data_infos)} of {len(lines)} samples (filtered missing npz).")
        return data_infos

    def get_gt_seg_map_by_filename(self, seg_map_filename):
        sparse = load_npz(seg_map_filename)  # (13, 262144)
        dense = sparse.toarray()  # shape: (13, 262144)
        
        if dense.shape[0] == 13:
            # One-hot → class index map
            dense = np.argmax(dense, axis=0).reshape(512, 512)  # shape: (512, 512)
        else:
            raise ValueError(f"Unexpected shape: {dense.shape} in {seg_map_filename}")

        return dense.astype(np.uint8)

    
    def get_gt_seg_map_by_idx(self, index):
        seg_map_path = self.img_infos[index]['ann_info']['seg_map']
        return self.get_gt_seg_map_by_filename(seg_map_path)

    def prepare_test_img(self, idx):
        results = dict(
            img_info=self.img_infos[idx]['img_info'],
            ann_info=self.img_infos[idx]['ann_info']
        )
        #print(f"[DEBUG] test results: {results}")
        return self.pipeline(results)


    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx]['img_info'])
        if 'ann_info' in self.img_infos[idx]:
            results['ann_info'] = self.img_infos[idx]['ann_info']
        return self.pipeline(results)
    
    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann_info']



