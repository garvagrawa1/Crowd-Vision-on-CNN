"""CrowdVision - real-time face detection & crowd counting with our own CNN.

Examples
  python detect.py                         # laptop webcam
  python detect.py --source 1              # external webcam
  python detect.py --source crowd.mp4      # video file
  python detect.py --source group.jpg      # single image (result saved to outputs/)
  python detect.py --capacity 5            # alert when more than 5 people are visible

Keys while running
  q / Esc   quit                 p / Space  pause / resume
  s         save screenshot      r          start / stop recording video
  + / -     raise / lower confidence threshold
  mouse     drag LEFT button to draw a counting zone, RIGHT click to clear it
"""
import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from detector import load_detector

GREEN, AMBER, RED, WHITE, DARK = (80, 200, 60), (0, 190, 255), (60, 60, 230), (255, 255, 255), (25, 25, 25)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------ tracking
class Track:
    def __init__(self, tid, box, score):
        self.id, self.box, self.score = tid, np.array(box, np.float32), score
        self.hits, self.miss = 1, 0


class FaceTracker:
    """Light IoU tracker: smooths boxes, removes 1-frame flickers, gives each face an ID."""

    def __init__(self, min_hits=2, max_miss=4, smooth=0.6):
        self.tracks, self.next_id = [], 1
        self.confirmed = 0          # faces that were seen for >= min_hits frames (real, not flicker)
        self.min_hits, self.max_miss, self.smooth = min_hits, max_miss, smooth

    @staticmethod
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0

    def update(self, dets):
        unmatched = list(range(len(dets)))
        for t in sorted(self.tracks, key=lambda t: -t.hits):
            best, best_iou = None, 0.3
            for j in unmatched:
                o = self.iou(t.box, dets[j][:4])
                if o > best_iou:
                    best, best_iou = j, o
            if best is None:
                t.miss += 1
            else:
                d = dets[best]
                t.box = self.smooth * t.box + (1 - self.smooth) * np.array(d[:4], np.float32)
                t.score, t.hits, t.miss = d[4], t.hits + 1, 0
                if t.hits == self.min_hits:
                    self.confirmed += 1
                unmatched.remove(best)
        for j in unmatched:
            self.tracks.append(Track(self.next_id, dets[j][:4], dets[j][4])); self.next_id += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return [t for t in self.tracks if t.hits >= self.min_hits]


# ------------------------------------------------------------------ drawing
def label(img, text, org, scale=0.55, color=WHITE, bg=DARK, thick=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + base), bg, -1)
    cv2.putText(img, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_box(img, box, text, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    L = max(8, (x2 - x1) // 5)                         # corner accents
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 4); cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 4)
    label(img, text, (x1, max(18, y1 - 2)), 0.45, DARK, color)


def crowd_level(n, capacity):
    if n > capacity:
        return "OVER CAPACITY", RED
    if n >= max(1, int(capacity * 0.7)):
        return "HIGH", AMBER
    if n >= max(1, int(capacity * 0.3)):
        return "MEDIUM", GREEN
    return "LOW", GREEN


def draw_hud(img, info):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), DARK, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, "CrowdVision", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "custom CNN (trained from scratch)", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    lvl, col = info["level"]
    cv2.putText(img, f"People: {info['count']}", (w // 2 - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    stats = f"FPS {info['fps']:.1f}  |  CNN {info['ms']:.0f} ms  |  thr {info['thr']:.2f}"
    cv2.putText(img, stats, (w - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Crowd: {lvl}   peak {info['peak']}   unique {info['unique']}", (w - 330, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if info.get("zone_count") is not None:
        label(img, f"In zone: {info['zone_count']}", (12, 90), 0.6, DARK, AMBER)
    if lvl == "OVER CAPACITY" and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 6)
        label(img, "ALERT: capacity exceeded", (w // 2 - 120, h - 20), 0.7, WHITE, RED, 2)
    if info.get("recording"):
        cv2.circle(img, (w - 20, 80), 8, RED, -1)
    if info.get("paused"):
        label(img, "PAUSED", (w // 2 - 40, h // 2), 1.0, WHITE, DARK, 2)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="CrowdVision real-time face counter (custom CNN)")
    ap.add_argument("--source", default="0", help="webcam index, video file, image file or folder")
    ap.add_argument("--model", default="crowdvision_cnn.pth")
    ap.add_argument("--threshold", type=float, default=0.99, help="face confidence (0.90-0.995); lower finds more faces but more false boxes")
    ap.add_argument("--min-face", type=int, default=48, help="smallest face size to search for (pixels, at --width)")
    ap.add_argument("--min-votes", type=int, default=2, help="windows needed to confirm a face")
    ap.add_argument("--width", type=int, default=640, help="processing width (smaller = faster)")
    ap.add_argument("--detect-every", type=int, default=1, help="run the CNN every N frames")
    ap.add_argument("--capacity", type=int, default=10, help="people limit for the crowd alert")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = auto)")
    ap.add_argument("--no-log", action="store_true", help="disable CSV count logging")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    det = load_detector(args.model, threshold=args.threshold, min_face=args.min_face, min_votes=args.min_votes)
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    print(f"Model loaded on {det.device}.  Press q to quit, s screenshot, r record, p pause, +/- threshold.")

    src = Path(args.source)
    if src.is_dir() or (src.suffix.lower() in IMAGE_EXT and src.exists()):
        run_images(det, src, args, out_dir)
        return
    run_stream(det, args, out_dir)


def resize_to(frame, width):
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return frame


def run_images(det, src, args, out_dir):
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = resize_to(img, max(args.width, 640) if img.shape[1] > args.width else img.shape[1])
        t0 = time.perf_counter(); dets = det.detect(img); ms = (time.perf_counter() - t0) * 1000
        for i, (x1, y1, x2, y2, s) in enumerate(dets, 1):
            draw_box(img, (x1, y1, x2, y2), f"#{i} {s:.2f}", GREEN)
        draw_hud(img, dict(count=len(dets), fps=1000 / ms, ms=ms, thr=det.threshold, peak=len(dets),
                           unique=len(dets), level=crowd_level(len(dets), args.capacity)))
        out = out_dir / f"result_{p.stem}.jpg"
        cv2.imwrite(str(out), img)
        print(f"{p.name}: {len(dets)} face(s)  ({ms:.0f} ms)  -> {out}")
        cv2.imshow("CrowdVision", img)
        if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def run_stream(det, args, out_dir):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source}")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "CrowdVision - press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    zone = {"rect": None, "start": None, "cur": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            zone["start"], zone["cur"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and zone["start"]:
            zone["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and zone["start"]:
            (x0, y0), zone["start"] = zone["start"], None
            if abs(x - x0) > 10 and abs(y - y0) > 10:
                zone["rect"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            zone["rect"] = None
    cv2.setMouseCallback(win, on_mouse)

    tracker = FaceTracker()
    counts = deque(maxlen=15)
    fps, ms, peak, frame_i = 0.0, 0.0, 0, 0
    paused, writer, frame = False, None, None
    tracks = []
    log_f = None
    if not args.no_log:
        (out_dir / "logs").mkdir(exist_ok=True)
        log_path = out_dir / "logs" / f"counts_{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_f = open(log_path, "w", newline=""); log_w = csv.writer(log_f)
        log_w.writerow(["timestamp", "people", "in_zone", "fps"]); last_log = 0
        print(f"Logging counts to {log_path}")
    t_prev = time.perf_counter()

    while True:
        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            frame = resize_to(raw, args.width)
            frame_i += 1
            if (frame_i - 1) % args.detect_every == 0:
                t0 = time.perf_counter(); dets = det.detect(frame); ms = (time.perf_counter() - t0) * 1000
                tracks = tracker.update(dets)
            counts.append(len(tracks))
        view = frame.copy()
        count = int(round(np.median(counts))) if counts else 0
        peak = max(peak, count)

        zone_count = None
        if zone["rect"]:
            zx1, zy1, zx2, zy2 = zone["rect"]
            cv2.rectangle(view, (zx1, zy1), (zx2, zy2), AMBER, 2)
            zone_count = 0
        for t in tracks:
            x1, y1, x2, y2 = t.box
            inside = False
            if zone["rect"]:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                inside = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                zone_count += inside
            draw_box(view, t.box, f"ID {t.id}  {t.score:.2f}", AMBER if inside else GREEN)
        if zone["start"] and zone["cur"]:
            cv2.rectangle(view, zone["start"], zone["cur"], AMBER, 1)

        now = time.perf_counter()
        if not paused:
            inst = 1.0 / max(1e-6, now - t_prev)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        t_prev = now
        draw_hud(view, dict(count=count, fps=fps, ms=ms, thr=det.threshold, peak=peak,
                            unique=tracker.confirmed, level=crowd_level(count, args.capacity),
                            zone_count=zone_count, recording=writer is not None, paused=paused))
        cv2.imshow(win, view)

        if writer is not None:
            writer.write(view)
        if log_f and not paused and now - last_log >= 1.0:
            log_w.writerow([datetime.now().isoformat(timespec="seconds"), count,
                            "" if zone_count is None else zone_count, round(fps, 1)]); last_log = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("p"), 32):
            paused = not paused
        elif key == ord("s"):
            (out_dir / "screenshots").mkdir(exist_ok=True)
            p = out_dir / "screenshots" / f"shot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(p), view); print(f"Saved {p}")
        elif key == ord("r"):
            if writer is None:
                p = out_dir / f"recording_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 15, (view.shape[1], view.shape[0]))
                print(f"Recording -> {p}")
            else:
                writer.release(); writer = None; print("Recording stopped")
        elif key in (ord("+"), ord("=")):
            det.threshold = min(0.999, det.threshold + 0.002)
        elif key in (ord("-"), ord("_")):
            det.threshold = max(0.5, det.threshold - 0.005)

    cap.release()
    if writer is not None:
        writer.release()
    if log_f:
        log_f.close()
    cv2.destroyAllWindows()
    print(f"Finished. Peak people count: {peak}, unique faces tracked: {tracker.confirmed}")


if __name__ == "__main__":
    main()
