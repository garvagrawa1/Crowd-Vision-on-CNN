"""Capture face-free photos of the demo location to reduce false detections.

Point the webcam at the room WITHOUT anyone in view (move around, change the
angle a little) and press SPACE to save a frame. 15-30 frames are plenty.
Then retrain:  python prepare_data.py  &&  python train.py
train.py will automatically mine "hard negatives" from these photos, so any
wall, poster or curtain that looked face-like is learned as non-face.

Keys: SPACE save frame | q quit
"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default="data/backgrounds")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    n = len(list(out.glob("*.jpg")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"NO FACES in view! SPACE = save ({n} saved)  q = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Collect backgrounds", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            p = out / f"bg_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(p), frame); n += 1; print("saved", p)
        elif k in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
