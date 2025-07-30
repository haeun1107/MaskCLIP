# -*- coding: utf-8 -*-

UNSEEN_CLASSES = [
    'spleen', 'gallbladder', 'stomach'
]


# 복사해온 로그를 문자열로 붙여넣기 (여기서는 예시 일부만 표시)
raw_result = """
|          background          | 98.91 | 99.73 | 99.17 |
|            spleen            | 80.03 | 85.79 | 92.26 |
|         kidney_right         | 73.17 | 78.52 | 91.48 |
|         kidney_left          | 54.37 | 61.19 | 82.99 |
|         gallbladder          | 37.35 | 42.52 | 75.41 |
|          esophagus           | 90.33 | 93.71 | 96.16 |
|            liver             | 78.65 | 81.85 | 95.26 |
|           stomach            | 64.81 | 71.37 | 87.57 |
|            aorta             | 63.21 | 74.99 | 80.09 |
|      inferior_vena_cava      | 40.13 | 47.87 | 71.26 |
| portal_vein_and_splenic_vein | 48.73 | 54.48 | 82.19 |
|           pancreas           |  0.0  |  0.0  |  nan  |
|     adrenal_gland_right      |  1.22 |  1.22 | 76.11 |
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
