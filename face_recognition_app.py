"""CrowdVision Face Recognition - who is in the crowd?

Uses TWO of our own CNNs, both trained from scratch (no YOLO, no pretrained models):
  1. CrowdVisionNet  (crowdvision_cnn.pth)  -> finds every face in the frame
  2. RecogNet        (recognizer_cnn.pth)   -> tells WHO each face is (or "Unknown")

Just run:      python face_recognition_app.py
and use the menu:
  1  Enroll a person   - webcam auto-captures ~60 photos of one face
  2  Train recognizer  - trains RecogNet on enrolled people (1-2 minutes on CPU)
  3  Live recognition  - CrowdVision screen with names, known/unknown count, attendance log
  4  List enrolled people

Or directly:   python face_recognition_app.py enroll --name Gauransh
               python face_recognition_app.py train
               python face_recognition_app.py run --capacity 5
"""
import argparse
import random
import sys
import time
import zipfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detector import load_detector
from detect import FaceTracker, draw_box, draw_hud, crowd_level, resize_to, label, GREEN, AMBER, RED, WHITE, DARK

PEOPLE_DIR = Path("data/people")
RECOG_PATH = Path("recognizer_cnn.pth")
SIZE = 64                    # recognizer input size
UNKNOWN = "Unknown"
LFW_FACE_BOX = (75, 80, 175, 180)


# ======================================================================= the recognition CNN
class RecogNet(nn.Module):
    """Small CNN: 64x64 face -> one score per enrolled person (+ 'Unknown')."""

    def __init__(self, n_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        # 64x64 -> 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 192))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_batch(crops):
    arr = np.stack(crops).astype(np.float32) / 255.0 - 0.5
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def face_crop(frame, box, size=SIZE, margin=0.12):
    """Square crop around a detected face (a little margin), resized for RecogNet."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + margin)
    H, W = frame.shape[:2]
    a, b, c, d = int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)
    pad = max(0, -a, -b, c - W, d - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        a, b, c, d = a + pad, b + pad, c + pad, d + pad
    crop = frame[b:d, a:c]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ======================================================================= 1. enroll
def open_camera(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/source {source}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def enroll(name, source="0", samples=60, width=640):
    name = name.strip().replace(" ", "_")
    if not name:
        raise SystemExit("Please give a name.")
    out = PEOPLE_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("*.jpg")))
    det = load_detector(threshold=0.99, min_face=60)
    cap = open_camera(source)
    win = f"Enroll {name} - look at the camera, turn your head slightly, q to stop"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    saved, frame_i, last_box = 0, 0, None
    print(f"Enrolling '{name}'. Only ONE face should be in view. Move your head a little: left, right, up, smile...")
    while saved < samples:
        ok, raw = cap.read()
        if not ok:
            break
        frame = resize_to(raw, width)
        frame_i += 1
        dets = det.detect(frame)
        view = frame.copy()
        msg, col = "", WHITE
        if len(dets) == 1:
            box = dets[0]
            moved = last_box is None or np.abs(np.array(box[:4]) - np.array(last_box[:4])).sum() > 6
            if frame_i % 3 == 0 and moved:
                crop = face_crop(frame, box, size=112)
                if crop is not None:
                    cv2.imwrite(str(out / f"{name}_{have + saved:04d}.jpg"), crop)
                    saved += 1; last_box = box
            draw_box(view, box[:4], f"{name}  {saved}/{samples}", GREEN)
            msg, col = "Capturing... move your head slowly", GREEN
        elif len(dets) == 0:
            msg, col = "No face found - come closer / more light", AMBER
        else:
            msg, col = "More than one face - only the person being enrolled please", RED
            for d in dets:
                draw_box(view, d[:4], "?", RED)
        bar_w = int(view.shape[1] * saved / samples)
        cv2.rectangle(view, (0, view.shape[0] - 10), (bar_w, view.shape[0]), GREEN, -1)
        label(view, msg, (10, 30), 0.6, DARK, col)
        cv2.imshow(win, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows()
    total = len(list(out.glob("*.jpg")))
    print(f"Saved {saved} new photos for '{name}' (total {total}) in {out}")
    if total < 20:
        print("Tip: at least 40 photos per person gives much better recognition.")
    return saved


# ======================================================================= 2. train
def unknown_faces(n=1500, seed=0):
    """Faces of strangers (LFW) for the 'Unknown' class, read straight from the dataset."""
    rng = random.Random(seed)
    crops = []
    folder = Path("datasets/lfw/lfw-deepfunneled")
    zpath = Path("datasets/lfw.zip")

    def take(img):
        x1, y1, x2, y2 = LFW_FACE_BOX
        j = rng.uniform(-6, 6); k = rng.uniform(-6, 6); s = rng.uniform(-8, 8)
        box = (x1 + j - s, y1 + k - s, x2 + j + s, y2 + k + s)
        c = face_crop(img, box)
        if c is not None:
            crops.append(c)

    if folder.exists():
        files = sorted(folder.glob("*/*.jpg")); rng.shuffle(files)
        for p in files[:n]:
            img = cv2.imread(str(p))
            if img is not None:
                take(img)
    elif zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            names = [m for m in z.namelist() if m.lower().endswith(".jpg")]
            rng.shuffle(names)
            for m in names[:n]:
                img = cv2.imdecode(np.frombuffer(z.read(m), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    take(img)
    else:
        print("WARNING: LFW dataset not found (datasets/lfw.zip). Training without an Unknown class -"
              " every face will be matched to someone. Run: python download_datasets.py")
    return crops


def augment(img, rng):
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    # small shift / zoom / rotation, like a person moving in front of the camera
    ang = rng.uniform(-12, 12); sc = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((SIZE / 2 + rng.uniform(-3, 3), SIZE / 2 + rng.uniform(-3, 3)), ang, sc)
    img = cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    img = img.astype(np.float32) * rng.uniform(0.65, 1.35) + rng.uniform(-30, 30)
    if rng.random() < 0.4:
        img = img * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 1.2))
    return img


def train(epochs=20, unknown_n=1200):
    people = sorted(p.name for p in PEOPLE_DIR.iterdir() if p.is_dir() and any(p.glob("*.jpg"))) if PEOPLE_DIR.exists() else []
    if not people:
        raise SystemExit("Nobody enrolled yet. First run:  python face_recognition_app.py enroll --name YourName")
    X, Y = [], []
    for ci, name in enumerate(people):
        for f in sorted((PEOPLE_DIR / name).glob("*.jpg")):
            img = cv2.imread(str(f))
            if img is not None:
                X.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)); Y.append(ci)
    classes = list(people)
    print("Loading stranger faces for the 'Unknown' class ...")
    unk = unknown_faces(unknown_n)
    if unk:
        classes.append(UNKNOWN)
        X += unk; Y += [len(classes) - 1] * len(unk)
    X, Y = np.stack(X), np.array(Y)
    counts = Counter(Y.tolist())
    print("Training data: " + ", ".join(f"{classes[c]}={counts[c]}" for c in range(len(classes))))

    # 85 / 15 split per class
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Y)); va_mask = np.zeros(len(Y), bool)
    for c in range(len(classes)):
        ci = idx[Y[idx] == c]; va_mask[ci[: max(1, int(0.15 * len(ci)))]] = True
    Xtr, Ytr, Xva, Yva = X[~va_mask], Y[~va_mask], X[va_mask], Y[va_mask]

    # balance classes: each enrolled person is sampled as often as the Unknown class
    weights = np.array([1.0 / counts[int(y)] for y in Ytr]); weights /= weights.sum()
    torch.manual_seed(0); prng = random.Random(0)
    model = RecogNet(len(classes))       # random initialisation - trained from scratch
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    steps = max(20, len(Ytr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps)
    best, t0 = -1, time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0
        for _ in range(steps):
            b = rng.choice(len(Ytr), 64, p=weights)
            xb = to_batch([augment(Xtr[i], prng) for i in b]); yb = torch.from_numpy(Ytr[b]).long()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval()
        with torch.no_grad():
            pred = model(to_batch(list(Xva))).argmax(1).numpy()
        per_class = [float((pred[Yva == c] == c).mean()) for c in range(len(classes))]
        bal_acc = float(np.mean(per_class))
        print(f"epoch {ep:02d}/{epochs}  loss {tot/steps:.3f}  validation accuracy (balanced) {bal_acc*100:.1f}%")
        if bal_acc >= best:
            best = bal_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": classes, "size": SIZE}, RECOG_PATH)
    print("\nPer-person validation accuracy:")
    ck = torch.load(RECOG_PATH, weights_only=False); model.load_state_dict(ck["model_state_dict"]); model.eval()
    with torch.no_grad():
        pred = model(to_batch(list(Xva))).argmax(1).numpy()
    for c, n in enumerate(classes):
        m = Yva == c
        print(f"  {n:15s} {(pred[m] == c).mean()*100:5.1f}%  ({m.sum()} test photos)")
    print(f"\nSaved {RECOG_PATH}  ({time.time()-t0:.0f}s).  Now run:  python face_recognition_app.py run")


# ======================================================================= 3. live recognition
class Recognizer:
    def __init__(self, path=RECOG_PATH, min_conf=0.70):
        if not Path(path).exists():
            raise SystemExit("recognizer_cnn.pth not found - enroll people and train first (menu options 1 and 2).")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.net = RecogNet(len(self.classes)); self.net.load_state_dict(ck["model_state_dict"]); self.net.eval()
        self.min_conf = min_conf

    @torch.no_grad()
    def predict(self, crops):
        if not crops:
            return []
        p = F.softmax(self.net(to_batch(crops)), 1).numpy()
        # average with the mirrored face for a steadier answer
        p = (p + F.softmax(self.net(to_batch([cv2.flip(c, 1) for c in crops])), 1).numpy()) / 2
        out = []
        for row in p:
            k = int(row.argmax())
            name = self.classes[k]
            if row[k] < self.min_conf:
                name = UNKNOWN
            out.append((name, float(row[k])))
        return out


def run(source="0", width=640, capacity=10, threshold=0.99, min_face=48, min_conf=0.70):
    det = load_detector(threshold=threshold, min_face=min_face)
    rec = Recognizer(min_conf=min_conf)
    known = [c for c in rec.classes if c != UNKNOWN]
    print(f"Recognising: {', '.join(known)}.  Keys: q quit, s screenshot, p pause, +/- detection threshold")
    cap = open_camera(source)
    win = "CrowdVision Face Recognition - press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    tracker = FaceTracker()
    votes = {}                                     # track id -> recent (name, conf)
    attendance = {}                                # name -> [first_seen, last_seen]
    out_dir = Path("outputs"); (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    counts = deque(maxlen=15); fps, ms, peak, paused, frame = 0.0, 0.0, 0, False, None
    tracks, t_prev = [], time.perf_counter()

    while True:
        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            frame = resize_to(raw, width)
            t0 = time.perf_counter()
            tracks = tracker.update(det.detect(frame))
            crops, owners = [], []
            for t in tracks:
                c = face_crop(frame, t.box)
                if c is not None:
                    crops.append(c); owners.append(t.id)
            for tid, res in zip(owners, rec.predict(crops)):
                votes.setdefault(tid, deque(maxlen=12)).append(res)
            ms = (time.perf_counter() - t0) * 1000
            counts.append(len(tracks))
        view = frame.copy()
        count = int(round(np.median(counts))) if counts else 0
        peak = max(peak, count)
        present, n_unknown = [], 0
        now_s = datetime.now().strftime("%H:%M:%S")
        for t in tracks:
            v = votes.get(t.id)
            if not v:
                continue
            names = Counter(n for n, _ in v)
            name, n_votes = names.most_common(1)[0]
            conf = np.mean([c for n, c in v if n == name])
            if name != UNKNOWN and n_votes < 3:       # need a few agreeing frames before naming someone
                name = UNKNOWN
            if name == UNKNOWN:
                n_unknown += 1
                draw_box(view, t.box, f"Unknown  ID {t.id}", RED)
            else:
                present.append(name)
                draw_box(view, t.box, f"{name}  {conf*100:.0f}%", GREEN)
                attendance.setdefault(name, [now_s, now_s])[1] = now_s

        tnow = time.perf_counter()
        if not paused:
            inst = 1.0 / max(1e-6, tnow - t_prev); fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        t_prev = tnow
        draw_hud(view, dict(count=count, fps=fps, ms=ms, thr=det.threshold, peak=peak, unique=len(attendance),
                            level=crowd_level(count, capacity), paused=paused))
        # side panel: who is here
        y = 92
        label(view, f"Known: {len(present)}   Unknown: {n_unknown}", (10, y), 0.55, DARK, AMBER); y += 26
        for n in sorted(set(present)):
            label(view, f"  {n}", (10, y), 0.5, DARK, GREEN); y += 22
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("p"), 32):
            paused = not paused
        elif key == ord("s"):
            (out_dir / "screenshots").mkdir(exist_ok=True)
            p = out_dir / "screenshots" / f"recog_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(p), view); print("Saved", p)
        elif key in (ord("+"), ord("=")):
            det.threshold = min(0.999, det.threshold + 0.002)
        elif key in (ord("-"), ord("_")):
            det.threshold = max(0.5, det.threshold - 0.005)

    cap.release(); cv2.destroyAllWindows()
    log = out_dir / "logs" / f"attendance_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(log, "w") as f:
        f.write("name,first_seen,last_seen\n")
        for n, (a, b) in sorted(attendance.items()):
            f.write(f"{n},{a},{b}\n")
    print(f"Finished. People recognised: {', '.join(sorted(attendance)) or 'none'}. Attendance saved to {log}")


# ======================================================================= menu
def list_people():
    if not PEOPLE_DIR.exists() or not any(PEOPLE_DIR.iterdir()):
        print("Nobody enrolled yet."); return
    for p in sorted(PEOPLE_DIR.iterdir()):
        if p.is_dir():
            print(f"  {p.name:20s} {len(list(p.glob('*.jpg')))} photos")
    print("Trained model:", "yes" if RECOG_PATH.exists() else "no (run option 2)")


def menu():
    while True:
        print("\n========  CrowdVision Face Recognition  ========")
        print(" 1  Enroll a person (webcam)")
        print(" 2  Train recognizer")
        print(" 3  Start live recognition")
        print(" 4  List enrolled people")
        print(" 0  Exit")
        ch = input("Choose: ").strip()
        try:
            if ch == "1":
                name = input("Person's name: ").strip()
                enroll(name)
                if input("Enroll another person? (y/n): ").strip().lower() != "y":
                    if input("Train the recognizer now? (y/n): ").strip().lower() == "y":
                        train()
            elif ch == "2":
                train()
            elif ch == "3":
                run()
            elif ch == "4":
                list_people()
            elif ch in ("0", "q"):
                break
        except SystemExit as e:
            print(e)


def main():
    if len(sys.argv) == 1:
        menu(); return
    ap = argparse.ArgumentParser(description="CrowdVision face recognition (own CNNs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--name", required=True); e.add_argument("--source", default="0")
    e.add_argument("--samples", type=int, default=60)
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=20)
    r = sub.add_parser("run"); r.add_argument("--source", default="0"); r.add_argument("--width", type=int, default=640)
    r.add_argument("--capacity", type=int, default=10); r.add_argument("--threshold", type=float, default=0.99)
    r.add_argument("--min-face", type=int, default=48)
    r.add_argument("--min-conf", type=float, default=0.70, help="below this the face is shown as Unknown")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "enroll":
        enroll(a.name, a.source, a.samples)
    elif a.cmd == "train":
        train(a.epochs)
    elif a.cmd == "run":
        run(a.source, a.width, a.capacity, a.threshold, a.min_face, a.min_conf)
    else:
        list_people()


if __name__ == "__main__":
    main()
