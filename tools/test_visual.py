import argparse
import os
import os.path as osp
import numpy as np
import cv2
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import math

# 🔸 MaskCLIP 모듈 등록 (레포 구조에 맞춰 import만 하면 됨)
import maskclip  # noqa: F401


# ---------------------- utils ----------------------
def _ensure_size(img, size_wh):
    """img을 (W,H)로 리사이즈(NEAREST for masks, LINEAR for images)."""
    W, H = size_wh
    interp = cv2.INTER_NEAREST if img.ndim == 2 or img.dtype != np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(img, (W, H), interpolation=interp)

def _add_title(img_bgr, title):
    """이미지 위에 타이틀 바(상단 32px) 붙이고 글자 그리기."""
    h, w = img_bgr.shape[:2]
    bar = np.zeros((32, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _safe_join(base, rel):
    if rel is None:
        return None
    if not base:
        return rel
    return rel if osp.isabs(rel) else osp.join(base, rel)

def _get_img_ann_paths(dataset, idx):
    """mmseg dataset의 img_infos에서 (이미지경로, 라벨경로) 추출."""
    info = dataset.img_infos[idx]
    # image
    rel_img = info.get('filename') or (info.get('img_info', {}) or {}).get('filename') or info.get('img')
    # ann
    rel_ann = None
    if isinstance(info.get('ann'), dict):
        rel_ann = info['ann'].get('seg_map')
    if rel_ann is None and isinstance(info.get('ann_info'), dict):
        rel_ann = info['ann_info'].get('seg_map')

    img_base = getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    ann_base = getattr(dataset, 'ann_dir', None)

    def _resolve(base, rel):
        if rel is None: return None
        if osp.isabs(rel): return rel
        if base and str(rel).startswith(str(base)): return rel
        return _safe_join(base, rel)

    return _resolve(img_base, rel_img), _resolve(ann_base, rel_ann)

def _to_2d_mask(mask, target_shape=None, img_shape=None):
    """1D/2D/(1,H,W) 라벨을 (H,W)로 안전 변환."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)
    if m.ndim == 2:
        return m.astype(np.uint8)
    if m.ndim == 1:
        N = m.size
        if target_shape is not None:
            H, W = map(int, target_shape)
        elif img_shape is not None:
            H, W = int(img_shape[0]), int(img_shape[1])
        else:
            raise AssertionError("Need target/img shape to reshape 1D mask")
        if N == H * W:
            mC = m.reshape((H, W), order='C')
            mF = m.reshape((H, W), order='F')
            def _stripe(a): return np.abs(np.diff(a,0)).sum()+np.abs(np.diff(a,1)).sum()
            return (mC if _stripe(mC) < _stripe(mF) else mF).astype(np.uint8)
        # fallback (근사)
        side = int(round(np.sqrt(N)))
        h = max(1, side)
        w = max(1, N // h)
        m2 = m[:h*w].reshape(h, w)
        return cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    raise AssertionError(f"mask must be 1D/2D/3D, got {m.shape}")

def _colorize(mask, palette, ignore_index=255):
    if mask is None: return None
    mask = mask.astype(np.int32)
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.uint8)
    K = len(palette)
    for cls_idx in np.unique(mask):
        if cls_idx == ignore_index or cls_idx < 0:  # 255/ -1 무시
            continue
        color = (0,0,0) if cls_idx >= K else palette[cls_idx]
        out[mask == cls_idx] = color
    return out

def _make_palette(n, scheme='bright', seed=0):
    base_bright = [
        (0,92,255),(0,255,255),(34,139,34),(255,0,0),(255,0,255),(255,105,180),
        (147,20,255),(60,179,113),(128,128,0),(0,215,255),(180,130,70),
        (203,192,255),(50,205,50),(139,0,0),(0,128,128),(128,0,128),(255,255,255)
    ]
    if scheme == 'random':
        rng = np.random.RandomState(seed)
        hsv = np.stack([
            rng.permutation(np.linspace(0,179,n,endpoint=False)),
            np.full(n,200), np.full(n,255)
        ], 1).astype(np.uint8).reshape(1,n,3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
        return [tuple(map(int,c)) for c in bgr]
    return [tuple(map(int, base_bright[i % len(base_bright)])) for i in range(n)]

def _legend_strip(width, class_names, palette, max_cols=6, pad=8, bg=(35,35,35)):
    if not class_names: return np.zeros((1,width,3), np.uint8)
    row_h = 26
    rows = (len(class_names) + max_cols - 1) // max_cols
    h = pad*2 + rows*row_h
    strip = np.full((h, width, 3), bg, np.uint8)
    swatch = 18; txt_h = 16
    col_w = max(140, width // max_cols)
    i, y = 0, pad
    for _ in range(rows):
        x = pad
        for _ in range(max_cols):
            if i >= len(class_names): break
            color = palette[i] if i < len(palette) else (200,200,200)
            x2 = min(x + swatch, width - pad)
            cv2.rectangle(strip, (x, y), (x2, y + swatch), color, -1)
            cv2.putText(strip, class_names[i], (x + swatch + 6, y + txt_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
            x += col_w; i += 1
        y += row_h
    return strip

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2, ignore_index=255):
    for cls_idx in np.unique(mask):
        if cls_idx == ignore_index or cls_idx < 0:
            continue
        binm = (mask == cls_idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            color = palette[cls_idx] if cls_idx < len(palette) else (255,255,255)
            cv2.drawContours(canvas_bgr, cnts, -1, color, thickness, lineType=cv2.LINE_AA)
    return canvas_bgr
# ---------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='MaskCLIP test/eval/visualize')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint (.pth)')
    parser.add_argument('--aug-test', action='store_true', help='Use Flip & MultiScaleAug if present')
    parser.add_argument('--out', help='save raw outputs (.pkl/.pickle)')
    parser.add_argument('--format-only', action='store_true', help='format results only')
    parser.add_argument('--eval', type=str, nargs='+', help='metrics e.g. mIoU mDice')
    parser.add_argument('--show', action='store_true', help='(unused) mmseg show')
    parser.add_argument('--show-dir', help='dir to save triptychs')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none','pytorch','slurm','mpi'], default='none')
    parser.add_argument('--opacity', type=float, default=0.6)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--vis-mode', choices=['overlay','mask'], default='overlay')
    # vis extras
    parser.add_argument('--palette', default='dataset', choices=['dataset','bright','random'])
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--legend-pos', default='outside-top', choices=['outside-top','outside-bottom'])
    parser.add_argument('--legend-cols', type=int, default=6)
    parser.add_argument('--outline', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # GT 마스킹 관련
    parser.add_argument('--mask-by-gt', action='store_true',
                        help='GT가 유효한 픽셀에서만 Prediction을 표시')
    parser.add_argument('--mask-ignore', type=int, default=255,
                        help='GT에서 무시(배경) 값 (기본 255)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Specify at least one of --out / --eval / --format-only / --show / --show-dir'

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a .pkl/.pickle')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # aug-test: MultiScaleFlipAug를 파이프라인에서 찾아서 세팅
    if args.aug_test:
        for step in cfg.data.test.pipeline:
            if step.get('type') == 'MultiScaleFlipAug':
                step.setdefault('flip', True)
                step.setdefault('img_scale', (512, 512))

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # data & loader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 메타 세이프가드: ckpt 메타 없으면 dataset 메타로 대체
    ckpt_meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    if not hasattr(model, 'CLASSES') or model.CLASSES is None:
        model.CLASSES = ckpt_meta.get('CLASSES', getattr(dataset, 'CLASSES', None))
    if not hasattr(model, 'PALETTE') or model.PALETTE is None:
        model.PALETTE = ckpt_meta.get('PALETTE', getattr(dataset, 'PALETTE', None))

    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    efficient_test = eval_kwargs.get('efficient_test', False)

    # inference
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, None, efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    print('test done')

    # 저장/평가
    if args.out:
        print(f'\n[IO] writing results to {args.out}')
        mmcv.dump(outputs, args.out)
    if args.format_only:
        dataset.format_results(outputs, **eval_kwargs)
    if args.eval:
        dataset.evaluate(outputs, args.eval, **eval_kwargs)

    # ------------------ 시각화 ------------------
    if not args.show_dir:
        return
    save_root = osp.abspath(args.show_dir)
    mmcv.mkdir_or_exist(save_root)
    print(f'[VIS] saving triptych to: {save_root}')

    # 팔레트/클래스
    palette = getattr(dataset, 'PALETTE', None) or getattr(model, 'PALETTE', None)
    if args.palette != 'dataset' or palette is None:
        num_classes = len(getattr(dataset, 'CLASSES', []) or [])
        palette = _make_palette(num_classes, scheme='random' if args.palette == 'random' else 'bright', seed=args.seed)
    class_names = list(getattr(dataset, 'CLASSES', []) or [])
    N = len(dataset)

    saved = 0
    for i in range(N):
        img_path, _ = _get_img_ann_paths(dataset, i)
        if not img_path or not osp.exists(img_path):
            continue

        # Input
        img = mmcv.imread(img_path)
        H, W = img.shape[:2]

        # Prediction
        pred = outputs[i]
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = np.asarray(pred).astype(np.uint8)
        if pred.ndim == 3 and pred.shape[0] == 1: pred = pred[0]
        if pred.ndim == 1:
            pred = _to_2d_mask(pred, target_shape=(H, W), img_shape=img.shape)
        elif pred.ndim == 2 and pred.shape != (H, W):
            pred = _ensure_size(pred, (W, H))

        pred_color = _colorize(pred, palette)
        pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0) \
                   if args.vis_mode == 'overlay' else pred_color

        # ---- Ground Truth 먼저 로드 (마스킹/윤곽선 등에 사용) ----
        gt = None
        try:
            gt = dataset.get_gt_seg_map_by_idx(i)
            if gt.shape != (H, W):
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        except Exception:
            gt = None

        # ---- Prediction을 GT로 마스킹 (클래스 픽셀만 보이게) ----
        if args.mask_by_gt and gt is not None:
            valid = (gt != args.mask_ignore)
            pc2 = pred_color.copy()
            pc2[~valid] = 0
            pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pc2, float(args.opacity), 0) \
                       if args.vis_mode == 'overlay' else pc2

        # GT 시각화
        if gt is not None:
            gt_color = _colorize(gt, palette)
            gt_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), gt_color, float(args.opacity), 0)
        else:
            gt_vis = np.zeros_like(img)

        # ---- Prediction을 GT로 마스킹 (화면용) ----
        if args.mask_by_gt and gt is not None:
            valid = (gt != args.mask_ignore)
            pc2 = pred_color.copy()
            pc2[~valid] = 0
            pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pc2, float(args.opacity), 0) \
                    if args.vis_mode == 'overlay' else pc2

        # 윤곽선용 라벨도 동일하게 마스킹
        pred_for_contour = pred.copy()
        if args.mask_by_gt and gt is not None:
            pred_for_contour[~valid] = args.mask_ignore
    
        # 윤곽선
        if args.outline:
            pred_vis = _draw_class_contours(pred_vis, pred_for_contour, palette,
                                            thickness=2, ignore_index=args.mask_ignore)
            if gt is not None:
                gt_vis = _draw_class_contours(gt_vis, gt, palette,
                                            thickness=2, ignore_index=args.mask_ignore)

        trip = cv2.hconcat([
            _add_title(img, 'Input'),
            _add_title(pred_vis, 'Prediction'),
            _add_title(gt_vis, 'Ground Truth'),
        ])

        if args.legend and class_names:
            strip = _legend_strip(trip.shape[1], class_names, palette, max_cols=args.legend_cols)
            trip = cv2.vconcat([strip, trip]) if args.legend_pos == 'outside-top' else cv2.vconcat([trip, strip])

        stem = osp.splitext(osp.basename(img_path))[0]
        mmcv.imwrite(trip, osp.join(save_root, f'{i:06d}_{stem}.png'))
        saved += 1

    print(f'[VIS] saved {saved}/{N} images at: {save_root}')


if __name__ == '__main__':
    main()
