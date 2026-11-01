"""Face detection engine shared by detect.py, evaluate.py and train.py.

Pipeline for one frame
  1. Image pyramid: resize the frame so that faces of every size become ~48 px
     somewhere in the pyramid (scale = 48 / face_size, shrinking by 0.79 each level).
  2. Run CrowdVisionNet on each pyramid level in ONE pass -> a probability map
     (one value per 48x48 window, windows every 8 px) + box offsets.
  3. Keep windows with face probability >= threshold, refine them with the
     predicted offsets and map them back to the original frame.
  4. Clustered NMS ("voting"): overlapping boxes are grouped, the group is
     replaced by its confidence-weighted average box, and groups supported by
     fewer than `min_votes` windows are dropped. A real face fires many
     neighbouring windows/scales; random textures usually fire only one.
     -> exactly one stable box per face, far fewer false positives.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import CrowdVisionNet, WINDOW, STRIDE

PAD = 12   # pixels of black border added around every pyramid level


def to_tensor(bgr_uint8):
    """HxWx3 uint8 (OpenCV BGR) -> 1x3xHxW float in [-0.5, 0.5]."""
    t = torch.from_numpy(np.ascontiguousarray(bgr_uint8)).permute(2, 0, 1).float()
    return (t / 255.0 - 0.5).unsqueeze(0)


def box_iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
    iom = inter / (np.minimum(area_a[:, None], area_b[None, :]) + 1e-9)  # intersection over min area
    return iou, iom


def nms(boxes, scores, iou_thr):
    """Classic greedy non-maximum suppression. Returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou, _ = box_iou_matrix(boxes[i:i + 1], boxes[order[1:]])
        order = order[1:][iou[0] < iou_thr]
    return np.array(keep, dtype=int)


def cluster_merge(boxes, scores, iou_thr=0.3, min_votes=2):
    """Group overlapping detections of SIMILAR size and fuse each group into one box."""
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    used = np.zeros(len(boxes), bool)
    out_b, out_s, out_v = [], [], []
    iou, _ = box_iou_matrix(boxes, boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        members = (~used) & (iou[i] >= iou_thr)
        members[i] = True
        used |= members
        w = scores[members][:, None]
        out_b.append((boxes[members] * w).sum(0) / w.sum())
        out_s.append(scores[members].max())
        out_v.append(int(members.sum()))
    out_b, out_s, out_v = np.array(out_b), np.array(out_s), np.array(out_v)
    keep = out_v >= min_votes
    out_b, out_s, out_v = out_b[keep], out_s[keep], out_v[keep]
    # Nested boxes: one face cannot contain another face.
    #  - big box around a smaller detected face  -> the big one is wrong (face + surroundings)
    #  - two boxes of similar size, mostly overlapping -> keep the more confident one
    if len(out_b) > 1:
        area = (out_b[:, 2] - out_b[:, 0]) * (out_b[:, 3] - out_b[:, 1])
        _, iom = box_iou_matrix(out_b, out_b)
        alive = np.ones(len(out_b), bool)
        for i in np.argsort(-out_s):
            if not alive[i]:
                continue
            for j in range(len(out_b)):
                if j == i or not alive[j] or iom[i, j] < 0.6:
                    continue
                if area[j] > 2.0 * area[i]:
                    alive[j] = False             # j is a big box around face i
                elif area[i] > 2.0 * area[j]:
                    if out_v[j] >= 2 and out_s[j] >= out_s[i] - 0.05:
                        alive[i] = False          # i is a big box around face j
                        break
                else:
                    alive[j] = False             # similar size duplicate, i is more confident
        out_b, out_s, out_v = out_b[alive], out_s[alive], out_v[alive]
    return out_b, out_s, out_v


class FaceDetector:
    def __init__(self, model_path="crowdvision_cnn.pth", device=None, threshold=0.99,
                 min_face=40, max_face=None, scale_factor=0.75, min_votes=2):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.net = CrowdVisionNet()
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.net.to(self.device).eval()
        self.threshold = threshold
        self.min_face = min_face
        self.max_face = max_face
        self.scale_factor = scale_factor
        self.min_votes = min_votes

    def scales_for(self, h, w):
        scales = []
        s = WINDOW / float(self.min_face)
        max_face = self.max_face or min(h, w)
        min_s = WINDOW / float(max_face)
        while min(h, w) * s >= WINDOW and s >= min_s * 0.95:
            scales.append(s)
            s *= self.scale_factor
        return scales

    @torch.no_grad()
    def raw_windows(self, frame, threshold=None):
        """All windows above threshold (before merging): boxes Nx4, scores N."""
        thr = self.threshold if threshold is None else threshold
        h, w = frame.shape[:2]
        all_b, all_s = [], []
        for s in self.scales_for(h, w):
            sw, sh = int(round(w * s)), int(round(h * s))
            img = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            # black border so faces partly outside the frame can still be found
            img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
            logits, reg = self.net(to_tensor(img).to(self.device))
            prob = F.softmax(logits, 1)[0, 1].cpu().numpy()
            ys, xs = np.where(prob >= thr)
            if len(ys) == 0:
                continue
            r = reg[0].cpu().numpy()[:, ys, xs].T * WINDOW
            x1 = xs * STRIDE - PAD; y1 = ys * STRIDE - PAD
            b = np.stack([x1, y1, x1 + WINDOW, y1 + WINDOW], 1).astype(np.float32) + r
            b /= s
            sc = prob[ys, xs]
            # drop windows that lie mostly in the black padding (outside the real frame)
            fw, fh = w, h
            ix = np.clip(b[:, 2], 0, fw) - np.clip(b[:, 0], 0, fw)
            iy = np.clip(b[:, 3], 0, fh) - np.clip(b[:, 1], 0, fh)
            inside = (ix * iy) / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + 1e-9)
            ok = inside >= 0.7
            b, sc = b[ok], sc[ok]
            if len(b) == 0:
                continue
            k = nms(b, sc, 0.6)               # light per-scale NMS keeps things fast
            all_b.append(b[k]); all_s.append(sc[k])
        if not all_b:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return np.concatenate(all_b), np.concatenate(all_s)

    def detect(self, frame):
        """Returns list of (x1, y1, x2, y2, score) in frame coordinates."""
        b, s = self.raw_windows(frame)
        b, s, v = cluster_merge(b, s, min_votes=self.min_votes)
        h, w = frame.shape[:2]
        dets = []
        for (x1, y1, x2, y2), sc in zip(b, s):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 > 8 and y2 - y1 > 8:
                dets.append((int(x1), int(y1), int(x2), int(y2), float(sc)))
        return dets


def load_detector(path=None, **kw):
    path = Path(path or "crowdvision_cnn.pth")
    if not path.exists():
        raise SystemExit(f"Model file '{path}' not found. Train first:  python train.py")
    return FaceDetector(str(path), **kw)
