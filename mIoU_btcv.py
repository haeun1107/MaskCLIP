# -*- coding: utf-8 -*-

UNSEEN_CLASSES = [
    'spleen', 'gallbladder', 'stomach'
]


# 복사해온 로그를 문자열로 붙여넣기 (여기서는 예시 일부만 표시)
raw_result = """
|          background          | 98.85 | 99.82 | 99.03 |
|            spleen            |  0.0  |  0.0  |  nan  |
|         kidney_right         |  51.4 | 79.16 | 59.44 |
|         kidney_left          | 66.31 | 72.86 | 88.06 |
|         gallbladder          |  0.0  |  0.0  |  nan  |
|          esophagus           |  91.8 |  94.8 | 96.66 |
|            liver             | 83.53 | 88.51 | 93.69 |
|           stomach            |  0.0  |  0.0  |  nan  |
|            aorta             | 63.18 | 81.21 |  74.0 |
|      inferior_vena_cava      | 54.37 | 60.59 | 84.13 |
| portal_vein_and_splenic_vein | 60.43 |  65.5 | 88.64 |
|           pancreas           | 20.16 | 51.34 | 24.92 |
|     adrenal_gland_right      | 36.69 | 45.57 | 65.32 |
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
