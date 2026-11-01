"""Honest evaluation of CrowdVision on crowd scenes it has never seen.

We build synthetic "crowd" images: faces of TEST-split people (never used for
training) are blended onto TEST-split background photos at random sizes and
positions. Because we placed every face ourselves, the exact ground-truth boxes
and the true people count are known, so we can measure:

  * Precision  - of the boxes drawn, how many are real faces
  * Recall     - of the real faces, how many were found
  * F1 score
  * Counting   - mean absolute count error and % of images counted exactly
  * Speed      - average CNN time per image on this computer
  * Precision-recall curve over confidence thresholds (results/pr_curve.png)

Usage:  python evaluate.py            (300 scenes, ~1-2 min)
        python evaluate.py --scenes 100 --threshold 0.9
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from detector import load_detector, box_iou_matrix, cluster_merge
from prepare_data import split_of, EXCLUDED_CALTECH, LFW_FACE_BOX

CONTEXT = (35, 25, 215, 235)   # region of an LFW photo pasted into the scene (face + hair + neck)


def make_scene(rng, faces, backs, w=640, h=480, max_faces=10):
    bg = cv2.imread(str(rng.choice(backs)))
    bg = cv2.resize(bg, (w, h))
    n = rng.randint(1, max_faces)
    gts = []
    for _ in range(n * 6):
        if len(gts) >= n:
            break
        face_px = rng.uniform(40, 120)                      # face width in the scene
        k = face_px / (LFW_FACE_BOX[2] - LFW_FACE_BOX[0])
        cw, ch = int((CONTEXT[2] - CONTEXT[0]) * k), int((CONTEXT[3] - CONTEXT[1]) * k)
        x, y = rng.randint(0, w - cw - 1), rng.randint(0, h - ch - 1)
        gx1 = x + (LFW_FACE_BOX[0] - CONTEXT[0]) * k; gy1 = y + (LFW_FACE_BOX[1] - CONTEXT[1]) * k
        g = np.array([gx1, gy1, gx1 + face_px, gy1 + face_px])
        if gts and box_iou_matrix(np.array([[x, y, x + cw, y + ch]]), np.array([b[1] for b in gts]))[1].max() > 0.15:
            continue
        img = cv2.imread(str(rng.choice(faces)))
        patch = cv2.resize(img[CONTEXT[1]:CONTEXT[3], CONTEXT[0]:CONTEXT[2]], (cw, ch), interpolation=cv2.INTER_AREA)
        mask = np.zeros((ch, cw), np.float32)               # soft oval so edges blend naturally
        cv2.ellipse(mask, (cw // 2, ch // 2), (int(cw * 0.46), int(ch * 0.48)), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, cw * 0.04))[..., None]
        roi = bg[y:y + ch, x:x + cw].astype(np.float32)
        bg[y:y + ch, x:x + cw] = (patch * mask + roi * (1 - mask)).astype(np.uint8)
        gts.append((g, np.array([x, y, x + cw, y + ch])))
    return bg, np.array([g for g, _ in gts])


def match(pred, gt, thr):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    iou, _ = box_iou_matrix(pred, gt)
    tp, used = 0, set()
    for i in np.argsort(-iou.max(1)):
        j = int(iou[i].argmax())
        if iou[i, j] >= thr and j not in used:
            used.add(j); tp += 1
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crowdvision_cnn.pth")
    ap.add_argument("--lfw", default="datasets/lfw/lfw-deepfunneled")
    ap.add_argument("--caltech", default="datasets/caltech101/101_ObjectCategories")
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--iou", type=float, default=0.4, help="IoU needed to count a box as correct")
    ap.add_argument("--min-votes", type=int, default=2)
    args = ap.parse_args()

    faces = [p for p in Path(args.lfw).glob("*/*.jpg") if split_of(p.parent.name) == 2]
    backs = [p for p in Path(args.caltech).glob("*/*.jpg")
             if p.parent.name not in EXCLUDED_CALTECH and split_of("scene_" + p.name + p.parent.name) == 2]
    if not faces or not backs:
        raise SystemExit("Datasets not found - run download_datasets.py first.")
    det = load_detector(args.model, threshold=args.threshold, min_face=36, min_votes=args.min_votes)
    out = Path("results"); (out / "eval_examples").mkdir(parents=True, exist_ok=True)

    rng = random.Random(123)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    pr = {t: [0, 0, 0] for t in thresholds}           # tp, n_pred, n_gt
    tp = npred = ngt = 0
    abs_err, exact, times = [], 0, []
    for i in range(args.scenes):
        img, gt = make_scene(rng, faces, backs)
        t0 = time.perf_counter()
        raw_b, raw_s = det.raw_windows(img, threshold=min(thresholds))
        times.append(time.perf_counter() - t0)
        for t in thresholds:
            k = raw_s >= t
            b, s, _ = cluster_merge(raw_b[k], raw_s[k], min_votes=args.min_votes)
            m = match(b, gt, args.iou)
            pr[t][0] += m; pr[t][1] += len(b); pr[t][2] += len(gt)
        b, s, _ = cluster_merge(raw_b[raw_s >= args.threshold], raw_s[raw_s >= args.threshold], min_votes=args.min_votes)
        m = match(b, gt, args.iou)
        tp += m; npred += len(b); ngt += len(gt)
        abs_err.append(abs(len(b) - len(gt))); exact += len(b) == len(gt)
        if i < 12:
            vis = img.copy()
            for g in gt:
                cv2.rectangle(vis, tuple(g[:2].astype(int)), tuple(g[2:].astype(int)), (255, 150, 0), 1)
            for bb, sc in zip(b, s):
                cv2.rectangle(vis, tuple(bb[:2].astype(int)), tuple(bb[2:].astype(int)), (60, 220, 60), 2)
            cv2.putText(vis, f"true {len(gt)} | predicted {len(b)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(out / "eval_examples" / f"scene_{i:02d}.jpg"), vis)

    P = tp / max(1, npred); R = tp / max(1, ngt); F1 = 2 * P * R / max(1e-9, P + R)
    report = (
        f"CrowdVision evaluation on {args.scenes} unseen synthetic crowd scenes (640x480, 1-10 faces each)\n"
        f"threshold {args.threshold}, IoU >= {args.iou}, min votes {args.min_votes}\n\n"
        f"True faces          : {ngt}\nDetected boxes      : {npred}\nCorrect detections  : {tp}\n"
        f"Precision           : {P*100:.1f}%\nRecall              : {R*100:.1f}%\nF1 score            : {F1*100:.1f}%\n"
        f"Count MAE           : {np.mean(abs_err):.2f} people per image\n"
        f"Exact count         : {exact/args.scenes*100:.1f}% of images\n"
        f"Avg CNN time        : {np.mean(times)*1000:.0f} ms per 640x480 image ({det.device})\n\n"
        "Threshold sweep (precision / recall):\n" +
        "".join(f"  {t:.2f}: P {pr[t][0]/max(1,pr[t][1])*100:5.1f}%  R {pr[t][0]/max(1,pr[t][2])*100:5.1f}%\n" for t in thresholds)
    )
    (out / "eval_metrics.txt").write_text(report)
    print(report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ps = [pr[t][0] / max(1, pr[t][1]) * 100 for t in thresholds]
        Rs = [pr[t][0] / max(1, pr[t][2]) * 100 for t in thresholds]
        plt.figure(figsize=(5, 4.2))
        plt.plot(Rs, Ps, marker="o", color="#3b6fd8")
        for t, x, y in zip(thresholds, Rs, Ps):
            plt.annotate(f"{t}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        plt.xlabel("Recall (%)"); plt.ylabel("Precision (%)"); plt.title("Precision-recall vs threshold")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=130)
    except ImportError:
        pass
    print("Saved results/eval_metrics.txt, results/pr_curve.png and results/eval_examples/")


if __name__ == "__main__":
    main()
