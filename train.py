"""Train CrowdVisionNet from scratch (random initialisation, no pretrained weights).

Stage 1  - learn face / non-face + box offsets from the 48x48 crops
           made by prepare_data.py.
Stage 2  - HARD NEGATIVE MINING: run the stage-1 detector over full face-free
           scene photos, collect every window it wrongly calls a face, add
           them to the negatives and fine-tune. This is the classic
           "bootstrapping" step used in Viola-Jones and MTCNN, and it is what
           removes most false boxes on walls / furniture / clothes.

Outputs
  crowdvision_cnn.pth                 trained weights (used by detect.py)
  results/training_log.csv            per-epoch loss / accuracy / precision / recall
  results/training_curves.png         plots for the report
  results/test_metrics.txt            final crop-level test metrics + confusion matrix

Usage
  python train.py                     # full training (~25-40 min on a laptop CPU)
  python train.py --epochs 4          # quick run to check everything works
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import CrowdVisionNet, count_parameters

EXCLUDED_CALTECH = {"Faces", "Faces_easy", "BACKGROUND_Google"}


# ---------------------------------------------------------------- augmentation
def augment(img, rng):
    """Photometric augmentation so the model copes with webcams, low light, blur."""
    img = img.astype(np.float32)
    if rng.random() < 0.8:                              # brightness / contrast
        img = img * rng.uniform(0.6, 1.4) + rng.uniform(-35, 35)
    if rng.random() < 0.5:                              # colour cast (white balance)
        img = img * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if rng.random() < 0.15:                             # grey-scale image
        g = img.mean(2, keepdims=True); img = np.repeat(g, 3, 2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:                             # low-resolution face
        s = rng.choice([16, 20, 24, 32])
        img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), (48, 48))
    if rng.random() < 0.2:                              # motion / focus blur
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.5, 1.2))
    if rng.random() < 0.2:                              # sensor noise
        img = np.clip(img + np.random.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img


def cut_edge(img, rng):
    """Simulate a face partly outside the camera frame (e.g. chin below the bottom edge).
    detector.py pads every frame with black, so such faces can still be found."""
    img = img.copy()
    band = rng.randint(6, 14)
    side = rng.choices(["bottom", "left", "right", "top"], weights=[5, 2, 2, 1])[0]
    if side == "bottom":
        img[-band:] = 0
    elif side == "top":
        img[:band] = 0
    elif side == "left":
        img[:, :band] = 0
    else:
        img[:, -band:] = 0
    return img


class CropDataset(Dataset):
    def __init__(self, x, y, r, train):
        self.x, self.y, self.r, self.train = x, y, r, train
        self.rng = random.Random()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.train:
            img = augment(img, self.rng)
            if self.y[i] == 1 and self.rng.random() < 0.2:
                img = cut_edge(img, self.rng)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0 - 0.5
        return t, int(self.y[i]), torch.from_numpy(self.r[i])


# ---------------------------------------------------------------- loss
def compute_loss(logits, reg, y, r, ohem=0.7):
    logits = logits.flatten(1)            # Nx2
    reg = reg.flatten(1)                  # Nx4
    cls_mask = y >= 0                     # positives + negatives
    reg_mask = y != 0                     # positives + part faces
    loss_cls = torch.zeros((), device=logits.device)
    if cls_mask.any():
        l = F.cross_entropy(logits[cls_mask], y[cls_mask], reduction="none")
        k = max(1, int(ohem * l.numel()))  # online hard example mining: learn from the hardest 70%
        loss_cls = l.topk(k).values.mean()
    loss_reg = torch.zeros((), device=logits.device)
    if reg_mask.any():
        loss_reg = F.smooth_l1_loss(reg[reg_mask], r[reg_mask], beta=0.1)
    return loss_cls + 0.5 * loss_reg, loss_cls, loss_reg


@torch.no_grad()
def evaluate_crops(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = tn = fn = 0
    reg_err, reg_n = 0.0, 0
    for x, y, r in loader:
        x, y, r = x.to(device), y.to(device), r.to(device)
        logits, reg = model(x)
        p = F.softmax(logits.flatten(1), 1)[:, 1]
        m = y >= 0
        pred = p[m] >= thr
        yt = y[m] == 1
        tp += (pred & yt).sum().item(); fp += (pred & ~yt).sum().item()
        tn += (~pred & ~yt).sum().item(); fn += (~pred & yt).sum().item()
        rm = y != 0
        if rm.any():
            reg_err += (reg.flatten(1)[rm] - r[rm]).abs().mean(1).sum().item(); reg_n += rm.sum().item()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn,
                box_err=reg_err / max(1, reg_n))


def run_epochs(model, train_ds, val_loader, device, epochs, lr, batch, log_rows, stage, best_path):
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * it / total)))
    best_f1 = -1
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); run = 0.0; n = 0
        for x, y, r in loader:
            x, y, r = x.to(device), y.to(device), r.to(device)
            logits, reg = model(x)
            loss, _, _ = compute_loss(logits, reg, y, r)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item(); n += 1
        m = evaluate_crops(model, val_loader, device)
        row = dict(stage=stage, epoch=ep, train_loss=run / n, val_acc=m["acc"], val_precision=m["precision"],
                   val_recall=m["recall"], val_f1=m["f1"], val_box_err=m["box_err"], seconds=time.time() - t0)
        log_rows.append(row)
        print(f"[{stage}] epoch {ep:02d}/{epochs}  loss {row['train_loss']:.4f}  val acc {m['acc']*100:.2f}%  "
              f"prec {m['precision']*100:.2f}%  recall {m['recall']*100:.2f}%  box err {m['box_err']:.3f}  "
              f"({row['seconds']:.0f}s)")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            save(model, best_path)
    model.load_state_dict(torch.load(best_path, weights_only=True)["model_state_dict"])
    return model


def save(model, path):
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "window": 48, "stride": 8, "classes": ["non_face", "face"],
                "note": "CrowdVisionNet trained from scratch"}, path)


def mine_hard_negatives(model_path, scene_files, per_image_cap=8, threshold=0.7, max_total=40000):
    """Run the current detector on face-free photos; every hit is a false positive."""
    from detector import FaceDetector
    det = FaceDetector(model_path, threshold=threshold, min_face=24, min_votes=1)
    crops = []
    for i, p in enumerate(scene_files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 480:
            f = 480 / max(h, w); img = cv2.resize(img, (int(w * f), int(h * f)))
        b, s = det.raw_windows(img)
        order = np.argsort(-s)[:per_image_cap]
        for (x1, y1, x2, y2) in b[order]:
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(img.shape[1], x2)), int(min(img.shape[0], y2))
            if x2 - x1 >= 12 and y2 - y1 >= 12:
                crops.append(cv2.resize(img[y1:y2, x1:x2], (48, 48), interpolation=cv2.INTER_AREA))
        if i % 1000 == 0:
            print(f"   mining {i}/{len(scene_files)} images, hard negatives found: {len(crops)}")
    if len(crops) > max_total:          # keep memory and fine-tuning time reasonable
        crops = random.Random(0).sample(crops, max_total)
    return crops


def plot_curves(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot"); return
    x = list(range(1, len(rows) + 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [r["train_loss"] for r in rows], marker="o", color="#3b6fd8")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch (all stages)"); ax[0].grid(alpha=.3)
    for key, lab, col in (("val_acc", "accuracy", "#3b6fd8"), ("val_precision", "precision", "#2a9d62"),
                          ("val_recall", "recall", "#d8823b")):
        ax[1].plot(x, [r[key] * 100 for r in rows], marker="o", label=lab, color=col)
    ax[1].set_title("Validation (unseen people)"); ax[1].set_xlabel("epoch (all stages)")
    ax[1].set_ylabel("%"); ax[1].legend(); ax[1].grid(alpha=.3)
    stage2 = [i + 1 for i, r in enumerate(rows) if r["stage"] == "hard-neg"]
    if stage2:
        for a in ax:
            a.axvline(stage2[0] - 0.5, ls="--", color="grey"); a.text(stage2[0] - 0.4, a.get_ylim()[1], " hard-neg mining", va="top", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/crops_48.npz")
    ap.add_argument("--caltech", default="datasets/caltech101/101_ObjectCategories")
    ap.add_argument("--extra-negatives", default="data/backgrounds")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--finetune-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--output", default="crowdvision_cnn.pth")
    ap.add_argument("--no-mining", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip stage 1 and continue from the existing --output weights (mining + fine-tune only)")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    Path("results").mkdir(exist_ok=True)

    if not Path(args.data).exists():
        raise SystemExit("Crops not found. Run:  python prepare_data.py")
    d = np.load(args.data)
    x, y, r, sp = d["x"], d["y"].astype(np.int64), d["r"], d["split"]
    tr, va, te = sp == 0, sp == 1, sp == 2
    print(f"Device: {device} | crops train {tr.sum()}  val {va.sum()}  test {te.sum()}")

    model = CrowdVisionNet().to(device)       # <- random initialisation: training from scratch
    print(f"CrowdVisionNet parameters: {count_parameters(model):,}")
    val_loader = DataLoader(CropDataset(x[va], y[va], r[va], False), batch_size=512)
    test_loader = DataLoader(CropDataset(x[te], y[te], r[te], False), batch_size=512)
    rows = []

    t_start = time.time()
    if args.resume and Path(args.output).exists():
        model.load_state_dict(torch.load(args.output, weights_only=True)["model_state_dict"])
        print(f"Resumed from {args.output} - skipping stage 1")
    else:
        model = run_epochs(model, CropDataset(x[tr], y[tr], r[tr], True), val_loader, device,
                         args.epochs, args.lr, args.batch_size, rows, "base", args.output)

    if not args.no_mining:
        scenes = sorted(p for p in Path(args.caltech).glob("*/*.jpg") if p.parent.name not in EXCLUDED_CALTECH)
        from prepare_data import split_of
        scenes = [p for p in scenes if split_of("scene_" + p.name + p.parent.name) == 0]   # train split only
        extra = Path(args.extra_negatives)
        if extra.exists():
            scenes += [p for p in extra.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"\nHard negative mining on {len(scenes)} face-free photos ...")
        hard = mine_hard_negatives(args.output, scenes)
        print(f"Hard negatives collected: {len(hard)}")
        if hard:
            hx = np.stack(hard)
            x2 = np.concatenate([x[tr], hx]); y2 = np.concatenate([y[tr], np.zeros(len(hx), np.int64)])
            r2 = np.concatenate([r[tr], np.zeros((len(hx), 4), np.float32)])
            model = run_epochs(model, CropDataset(x2, y2, r2, True), val_loader, device,
                               args.finetune_epochs, args.lr / 5, args.batch_size, rows, "hard-neg", args.output)

    if rows:
        log = Path("results/training_log.csv")
        old = list(csv.DictReader(open(log))) if (args.resume and log.exists()) else []
        rows = [dict(r, **{k: float(v) for k, v in r.items() if k not in ("stage", "epoch")}) for r in old] + rows
        with open(log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        plot_curves(rows, "results/training_curves.png")

    m = evaluate_crops(model, test_loader, device)
    report = (
        "CrowdVisionNet - crop-level TEST results (people never seen in training)\n"
        f"Accuracy : {m['acc']*100:.2f}%\nPrecision: {m['precision']*100:.2f}%\n"
        f"Recall   : {m['recall']*100:.2f}%\nF1 score : {m['f1']*100:.2f}%\n"
        f"Mean box offset error: {m['box_err']:.3f} (fraction of window size)\n\n"
        "Confusion matrix            predicted face   predicted non-face\n"
        f"  actual face               {m['tp']:>10}         {m['fn']:>10}\n"
        f"  actual non-face           {m['fp']:>10}         {m['tn']:>10}\n"
        f"\nTotal training time: {(time.time()-t_start)/60:.1f} min on {device}\n"
    )
    Path("results/test_metrics.txt").write_text(report)
    print("\n" + report)
    print(f"Saved model -> {args.output}   (plots + logs in results/)")


if __name__ == "__main__":
    main()
