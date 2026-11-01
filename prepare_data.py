"""Build the 48x48 training crops for CrowdVisionNet.

Sources (both public, downloaded by download_datasets.py):
  * LFW (Labeled Faces in the Wild, deep-funneled): ~13k photos of ~5.7k people.
    Every photo is 250x250 and aligned so the main face sits in a known box.
  * Caltech-101 object categories (the face categories are excluded): used as
    "everything that is not a face" (background / objects / scenes).

Crop types (same idea as the MTCNN paper):
  positive  IoU >= 0.65 with the face box   -> label face, + box offsets
  part      0.40 <= IoU < 0.65              -> box offsets only (teaches localisation)
  negative  IoU < 0.30 (hair, shoulders, half faces, scenery) -> label non-face

Identities are split into train / validation / test BY PERSON, so the model is
never evaluated on a person it trained on.

Output: datasets/crops_48.npz  (uint8 images + labels + offsets + split)
"""
import argparse
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

S = 48
LFW_FACE_BOX = (75, 80, 175, 180)          # face location in every 250x250 LFW image
EXCLUDED_CALTECH = {"Faces", "Faces_easy", "BACKGROUND_Google"}


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def split_of(name):
    """Deterministic person-level split: 80% train, 10% val, 10% test."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return 2 if h == 0 else (1 if h == 1 else 0)    # 0=train 1=val 2=test


def crop(img, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return cv2.resize(img[y1:y2, x1:x2], (S, S), interpolation=cv2.INTER_AREA)


def offsets(box, gt):
    s = box[2] - box[0]
    return [(gt[i] - box[i]) / s for i in range(4)]


def lfw_samples(img, rng, n_pos, n_part, n_neg):
    h, w = img.shape[:2]
    gt = LFW_FACE_BOX
    gs = gt[2] - gt[0]
    out = []
    want = {1: n_pos, -1: n_part, 0: n_neg}
    tries = 0
    while any(v > 0 for v in want.values()) and tries < 400:
        tries += 1
        kind = rng.choice([k for k, v in want.items() if v > 0])
        if kind == 0:
            size = gs * rng.uniform(0.5, 1.6)
            cx = (gt[0] + gt[2]) / 2 + rng.uniform(-0.75, 0.75) * size
            cy = (gt[1] + gt[3]) / 2 + rng.uniform(-0.75, 0.75) * size
        else:
            spread = 0.12 if kind == 1 else 0.3
            size = gs * rng.uniform(0.85, 1.2) if kind == 1 else gs * rng.uniform(0.7, 1.4)
            cx = (gt[0] + gt[2]) / 2 + rng.uniform(-spread, spread) * size
            cy = (gt[1] + gt[3]) / 2 + rng.uniform(-spread, spread) * size
        box = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h or size < 20:
            continue
        o = iou(box, gt)
        # negatives from LFW must still touch the main face (hair, half-face, shoulders);
        # crops far away could contain *other* people standing in the background.
        label = 1 if o >= 0.65 else (-1 if 0.4 <= o < 0.65 else (0 if 0.03 < o < 0.3 else None))
        if label is None or want[label] <= 0:
            continue
        want[label] -= 1
        out.append((crop(img, box), label, offsets(box, gt) if label != 0 else [0, 0, 0, 0]))
    return out


def background_samples(img, rng, n):
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        smax = min(h, w)
        if smax < 24:
            break
        size = rng.uniform(24, smax)
        x, y = rng.uniform(0, w - size), rng.uniform(0, h - size)
        out.append((crop(img, (x, y, x + size, y + size)), 0, [0, 0, 0, 0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="datasets/lfw/lfw-deepfunneled")
    ap.add_argument("--caltech", default="datasets/caltech101/101_ObjectCategories")
    ap.add_argument("--extra-negatives", default="data/backgrounds",
                    help="folder of face-free photos of YOUR room (optional, see collect_backgrounds.py)")
    ap.add_argument("--out", default="datasets/crops_48.npz")
    ap.add_argument("--pos-per-face", type=int, default=2)
    ap.add_argument("--part-per-face", type=int, default=1)
    ap.add_argument("--neg-per-face", type=int, default=2)
    ap.add_argument("--neg-per-scene", type=int, default=12)
    args = ap.parse_args()

    rng = random.Random(0)
    X, Y, R, SPLIT = [], [], [], []

    def add(samples, split):
        for im, lab, off in samples:
            X.append(im); Y.append(lab); R.append(off); SPLIT.append(split)

    lfw = sorted(Path(args.lfw).glob("*/*.jpg"))
    if not lfw:
        raise SystemExit(f"No LFW images in {args.lfw}. Run: python download_datasets.py")
    print(f"LFW images: {len(lfw)}")
    for i, p in enumerate(lfw):
        img = cv2.imread(str(p))
        if img is None:
            continue
        sp = split_of(p.parent.name)
        add(lfw_samples(img, rng, args.pos_per_face, args.part_per_face, args.neg_per_face), sp)
        # mirrored copy -> more variety, labels mirrored correctly
        flip = cv2.flip(img, 1)
        if rng.random() < 0.5:
            add(lfw_samples(flip, rng, 1, 0, 0), sp)
        if i % 2000 == 0:
            print(f"  LFW {i}/{len(lfw)}  crops so far: {len(X)}")

    scenes = [p for p in sorted(Path(args.caltech).glob("*/*.jpg")) if p.parent.name not in EXCLUDED_CALTECH]
    print(f"Caltech scene/object images (no faces): {len(scenes)}")
    for i, p in enumerate(scenes):
        img = cv2.imread(str(p))
        if img is None:
            continue
        sp = split_of("scene_" + p.name + p.parent.name)
        add(background_samples(img, rng, args.neg_per_scene), sp)

    extra = Path(args.extra_negatives)
    if extra.exists():
        files = [p for p in extra.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
        for p in files:
            img = cv2.imread(str(p))
            if img is not None:
                add(background_samples(img, rng, 300), 0)
        print(f"Extra background photos used: {len(files)}")

    X = np.stack(X).astype(np.uint8)
    Y = np.array(Y, np.int8)
    R = np.array(R, np.float32)
    SPLIT = np.array(SPLIT, np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=X, y=Y, r=R, split=SPLIT)
    for name, k in (("positive", 1), ("part", -1), ("negative", 0)):
        print(f"{name:9s}: {int((Y == k).sum()):7d}")
    print(f"train/val/test crops: {[(SPLIT == s).sum() for s in (0, 1, 2)]}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
