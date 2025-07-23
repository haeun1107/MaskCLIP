# -*- coding: utf-8 -*-

UNSEEN_CLASSES = [
    'cow', 'motorbike', 'sofa', 'cat', 'boat', 
    'fence', 'bird', 'tvmonitor', 'keyboard', 'aeroplane'
]


# 복사해온 로그를 문자열로 붙여넣기 (여기서는 예시 일부만 표시)
raw_result = """
|          background          | 98.85 | 99.72 | 99.12 |
|            spleen            | 78.83 | 84.82 | 91.77 |
|         kidney_right         | 71.61 | 77.99 | 89.74 |
|         kidney_left          | 48.39 | 53.37 | 83.83 |
|         gallbladder          |  38.5 |  43.2 | 77.96 |
|          esophagus           | 89.77 | 93.44 | 95.81 |
|            liver             | 74.82 | 78.29 |  94.4 |
|           stomach            |  64.9 | 70.23 | 89.52 |
|            aorta             | 62.71 | 73.93 | 80.52 |
|      inferior_vena_cava      |  39.3 | 47.04 | 70.49 |
| portal_vein_and_splenic_vein | 49.21 | 56.45 | 79.33 |
|           pancreas           |  0.0  |  0.0  |  nan  |
|     adrenal_gland_right      |  0.0  |  0.0  |  nan  |
|      adrenal_gland_left      |  nan  |  nan  |  nan  |
"""

def parse_class_iou(raw_text):
    import re
    iou_map = {}
    pattern = r'\|\s+([\w]+[\w\s]*)\s+\|\s+([\d.]+)\s+\|'
    for match in re.finditer(pattern, raw_text):
        class_name = match.group(1).strip()
        iou_value = float(match.group(2))
        iou_map[class_name] = iou_value
    return iou_map

def parse_class_iou(raw_text):
    import re
    iou_map = {}
    pattern = r'\|\s+([\w]+[\w\s]*)\s+\|\s+([\d.]+)\s+\|'
    for match in re.finditer(pattern, raw_text):
        class_name = match.group(1).strip()
        iou_value = float(match.group(2))
        iou_map[class_name] = iou_value
    return iou_map

def compute_metrics(iou_map):
    unseen_ious = [iou_map[c] for c in UNSEEN_CLASSES if c in iou_map]
    all_ious = [v for v in iou_map.values() if not str(v).lower() == 'nan']
    seen_ious = [iou_map[c] for c in iou_map if c not in UNSEEN_CLASSES and not str(iou_map[c]).lower() == 'nan']

    mIoU_U = sum(unseen_ious) / len(unseen_ious) if unseen_ious else float('nan')
    mIoU_S = sum(seen_ious) / len(seen_ious) if seen_ious else float('nan')
    mIoU = sum(all_ious) / len(all_ious) if all_ious else float('nan')
    hIoU = (
        2 * mIoU_U * mIoU_S / (mIoU_U + mIoU_S)
        if not any(map(lambda x: x != x, [mIoU_U, mIoU_S])) and (mIoU_U + mIoU_S) > 0
        else float('nan')
    )

    return mIoU_U, mIoU_S, mIoU, hIoU, len(seen_ious), len(unseen_ious), len(all_ious)

if __name__ == "__main__":
    iou_map = parse_class_iou(raw_result)
    mIoU_U, mIoU_S, mIoU, hIoU, n_seen, n_unseen, n_all = compute_metrics(iou_map)

    print(f"✅ # Seen classes   : {n_seen}")
    print(f"✅ # Unseen classes : {n_unseen}")
    print(f"✅ # Total classes  : {n_all}")
    print()
    print(f"📊 mIoU(U) (Unseen) : {mIoU_U:.2f}")
    print(f"📊 mIoU(S) (Seen)   : {mIoU_S:.2f}")
    print(f"📊 mIoU    (All)    : {mIoU:.2f}")
    print(f"📊 hIoU             : {hIoU:.2f}")
