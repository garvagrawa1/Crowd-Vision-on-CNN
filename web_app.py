"""CrowdVision web console - our own CNNs behind a browser dashboard.

Run:   python web_app.py
Open:  http://localhost:8000   (opens automatically)

Everything runs locally on this laptop:
  * CrowdVisionNet (crowdvision_cnn.pth) finds faces  - trained from scratch
  * RecogNet (recognizer_cnn.pth) recognises enrolled people - trained from scratch
No YOLO, no pretrained models, no internet needed (only Python's built-in web server).
"""
import argparse
import csv
import io
import json
import shutil
import threading
import time
import webbrowser
from collections import Counter, deque
from contextlib import redirect_stdout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from detector import load_detector
from detect import FaceTracker, resize_to
import face_recognition_app as fr

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = ROOT / "outputs"
STATS_FILE = ROOT / "data" / "registry_stats.json"

# colours (BGR) used for boxes drawn on the frames
KNOWN = (90, 190, 40)       # green  - recognised person
UNKNOWN = (52, 160, 232)    # amber  - face, not recognised
PLAIN = (224, 95, 76)       # indigo - face (recognition off)


def put_label(img, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_face(img, box, text, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    L = max(8, (x2 - x1) // 5)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 4)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 4)
    if text:
        put_label(img, text, x1, y1 - 2, color)


# ====================================================================== engine
class Engine:
    """Owns the camera, both CNNs, the tracker and all statistics."""

    def __init__(self, width=640):
        self.width = width
        self.lock = threading.Lock()
        self.det = load_detector(str(ROOT / "crowdvision_cnn.pth"), threshold=0.99, min_face=48)
        self.rec = None
        self.load_recognizer()
        self.settings = dict(threshold=0.99, capacity=8, alerts=True, logging=True,
                             show_conf=True, recognition=True)
        self.running = False
        self.source = "0"
        self.cap = None
        self.thread = None
        self.jpeg = None                     # latest annotated frame
        self.reset_session()
        self.enroll = None                   # dict while enrolling a person
        self.train_state = {"running": False, "log": [], "done": False, "error": None}
        self.reg_stats = self.load_reg_stats()

    # ---------------------------------------------------------- helpers
    def load_recognizer(self):
        try:
            self.rec = fr.Recognizer(ROOT / fr.RECOG_PATH) if (ROOT / fr.RECOG_PATH).exists() else None
        except SystemExit:
            self.rec = None

    def reset_session(self):
        self.tracker = FaceTracker()
        self.votes = {}
        self.track_names = {}
        self.counts = deque(maxlen=15)
        self.count = 0
        self.peak = 0
        self.conf_sum, self.conf_n = 0.0, 0
        self.detections_logged = 0
        self.fps, self.ms = 0.0, 0.0
        self.started_at = None
        self.history = []                    # [{time, count, source}]
        self.faces = []                      # current faces for the page
        self.last_sample = 0

    def load_reg_stats(self):
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            return {}

    def save_reg_stats(self):
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(self.reg_stats, indent=1))

    # ---------------------------------------------------------- camera control
    def start(self, source=None):
        with self.lock:
            if self.running:
                return True, "already running"
            self.source = str(source if source is not None else self.source)
            try:
                self.cap = fr.open_camera(self.source)
            except SystemExit as e:
                return False, str(e)
            self.reset_session()
            self.started_at = time.time()
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
            return True, "started"

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.enroll = None

    # ---------------------------------------------------------- per-frame work
    def recognise(self, frame, tracks):
        if not (self.settings["recognition"] and self.rec):
            return {}
        crops, owners = [], []
        for t in tracks:
            c = fr.face_crop(frame, t.box)
            if c is not None:
                crops.append(c); owners.append(t.id)
        for tid, res in zip(owners, self.rec.predict(crops)):
            self.votes.setdefault(tid, deque(maxlen=12)).append(res)
        names = {}
        for t in tracks:
            v = self.votes.get(t.id)
            if not v:
                continue
            name, n = Counter(x for x, _ in v).most_common(1)[0]
            conf = float(np.mean([c for x, c in v if x == name]))
            if name != fr.UNKNOWN and n >= 3:
                names[t.id] = (name, conf)
                if self.track_names.get(t.id) != name:          # new sighting of this person
                    self.track_names[t.id] = name
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st = self.reg_stats.setdefault(name, {"times_seen": 0, "first_seen": now, "last_seen": now})
                    st["times_seen"] += 1
                    st["last_seen"] = now
                    self.save_reg_stats()
                else:
                    self.reg_stats.get(name, {})["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                names[t.id] = (fr.UNKNOWN, conf)
        return names

    def do_enroll(self, frame, dets, view):
        e = self.enroll
        if len(dets) == 1:
            box = dets[0]
            moved = e["last"] is None or np.abs(np.array(box[:4]) - np.array(e["last"][:4])).sum() > 6
            e["frame"] += 1
            if e["frame"] % 3 == 0 and moved:
                crop = fr.face_crop(frame, box, size=112)
                if crop is not None:
                    cv2.imwrite(str(e["dir"] / f"{e['name']}_{e['start'] + e['saved']:04d}.jpg"), crop)
                    e["saved"] += 1; e["last"] = box
            draw_face(view, box[:4], f"ENROLLING {e['name']}  {e['saved']}/{e['target']}", KNOWN)
            e["msg"] = "Capturing - move your head slowly"
        elif not dets:
            e["msg"] = "No face found - come closer / add light"
        else:
            e["msg"] = "Only the person being enrolled should be in view"
            for d in dets:
                draw_face(view, d[:4], "?", (60, 60, 230))
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, h - 8), (int(w * e["saved"] / e["target"]), h), KNOWN, -1)
        if e["saved"] >= e["target"]:
            e["done"] = True

    def loop(self):
        prev = time.perf_counter()
        while self.running:
            ok, raw = self.cap.read()
            if not ok:
                if not str(self.source).isdigit():                 # video file finished: loop it
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            frame = resize_to(raw, self.width)
            view = frame.copy()
            self.det.threshold = self.settings["threshold"]
            t0 = time.perf_counter()
            dets = self.det.detect(frame)
            if self.enroll and not self.enroll.get("done"):
                self.do_enroll(frame, dets, view)
                tracks, names = [], {}
            else:
                tracks = self.tracker.update(dets)
                names = self.recognise(frame, tracks)
            self.ms = (time.perf_counter() - t0) * 1000

            faces = []
            for t in tracks:
                name, conf = names.get(t.id, (None, None))
                if name and name != fr.UNKNOWN:
                    col, label = KNOWN, f"{name}" + (f" {conf*100:.0f}%" if self.settings["show_conf"] else "")
                elif name == fr.UNKNOWN:
                    col, label = UNKNOWN, "Unknown" + (f" {t.score:.2f}" if self.settings["show_conf"] else "")
                else:
                    col, label = PLAIN, (f"Face {t.score:.2f}" if self.settings["show_conf"] else "")
                draw_face(view, t.box, label, col)
                faces.append({"id": t.id, "name": name, "score": round(float(t.score), 3)})
                self.conf_sum += float(t.score); self.conf_n += 1
            if not self.enroll:
                self.counts.append(len(tracks))
            count = int(round(np.median(self.counts))) if self.counts else 0
            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - prev); prev = now
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.count = count
                self.peak = max(self.peak, count)
                self.faces = faces
                if ok:
                    self.jpeg = buf.tobytes()
                if self.settings["logging"] and time.time() - self.last_sample >= 2 and not self.enroll:
                    self.last_sample = time.time()
                    self.history.append({"time": datetime.now().strftime("%H:%M:%S"), "count": count,
                                         "source": "Live" if str(self.source).isdigit() else "Video"})
                    self.history = self.history[-500:]
                    self.detections_logged += count

    # ---------------------------------------------------------- photo analyzer
    def analyze_image(self, data):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.shape[1] > 1280:
            img = resize_to(img, 1280)
        self.det.threshold = self.settings["threshold"]
        dets = self.det.detect(img)
        crops = [fr.face_crop(img, d) for d in dets]
        preds = self.rec.predict([c for c in crops if c is not None]) if (self.settings["recognition"] and self.rec and dets) else []
        out, faces = img.copy(), []
        for i, d in enumerate(dets):
            name, conf = preds[i] if i < len(preds) else (None, None)
            if name and name != fr.UNKNOWN:
                col, label = KNOWN, name + (f" {conf*100:.0f}%" if self.settings["show_conf"] else "")
            elif name == fr.UNKNOWN:
                col, label = UNKNOWN, "Unknown" + (f" {d[4]:.2f}" if self.settings["show_conf"] else "")
            else:
                col, label = PLAIN, (f"Face {d[4]:.2f}" if self.settings["show_conf"] else "")
            draw_face(out, d[:4], label, col)
            faces.append({"name": name, "score": round(d[4], 3)})
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.history.append({"time": datetime.now().strftime("%H:%M:%S"), "count": len(dets), "source": "Photo"})
            self.detections_logged += len(dets)
            self.peak = max(self.peak, len(dets))
        return buf.tobytes(), faces

    # ---------------------------------------------------------- registry
    def people(self):
        rows = []
        if fr.PEOPLE_DIR.exists():
            known = set(self.rec.classes) if self.rec else set()
            for d in sorted(p for p in (ROOT / fr.PEOPLE_DIR).iterdir() if p.is_dir()):
                info = {}
                try:
                    info = json.loads((d / "info.json").read_text())
                except Exception:
                    pass
                st = self.reg_stats.get(d.name, {})
                rows.append({"name": d.name, "tag": info.get("tag", ""), "category": info.get("category", ""),
                             "notes": info.get("notes", ""), "photos": len(list(d.glob("*.jpg"))),
                             "trained": d.name in known, "times_seen": st.get("times_seen", 0),
                             "first_seen": st.get("first_seen", ""), "last_seen": st.get("last_seen", "")})
        return rows

    def start_enroll(self, name, tag, category, notes, target=60):
        name = "".join(ch for ch in name.strip().replace(" ", "_") if ch.isalnum() or ch in "_-")
        if not name:
            return False, "Please type a name"
        if not self.running:
            ok, msg = self.start()
            if not ok:
                return False, msg
        d = ROOT / fr.PEOPLE_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "info.json").write_text(json.dumps({"tag": tag, "category": category, "notes": notes}))
        self.enroll = {"name": name, "dir": d, "start": len(list(d.glob("*.jpg"))), "saved": 0,
                       "target": target, "frame": 0, "last": None, "msg": "Starting...", "done": False}
        return True, "enrolling"

    def delete_person(self, name):
        d = ROOT / fr.PEOPLE_DIR / name
        if d.is_dir() and d.parent == (ROOT / fr.PEOPLE_DIR):
            shutil.rmtree(d)
        self.reg_stats.pop(name, None)
        self.save_reg_stats()

    def start_training(self):
        if self.train_state["running"]:
            return False
        self.train_state = {"running": True, "log": ["Training RecogNet from scratch..."], "done": False, "error": None}

        class LogWriter(io.TextIOBase):
            def __init__(s, st):
                s.st = st
            def write(s, txt):
                for line in txt.splitlines():
                    if line.strip():
                        s.st["log"].append(line.strip())
                        s.st["log"] = s.st["log"][-40:]
                return len(txt)

        def job():
            try:
                with redirect_stdout(LogWriter(self.train_state)):
                    fr.train()
                self.load_recognizer()
                self.votes.clear(); self.track_names.clear()
                self.train_state["log"].append("Done - recognition model reloaded.")
            except SystemExit as e:
                self.train_state["error"] = str(e)
            except Exception as e:                              # show any problem on the page
                self.train_state["error"] = f"{type(e).__name__}: {e}"
            finally:
                self.train_state["running"] = False
                self.train_state["done"] = True

        threading.Thread(target=job, daemon=True).start()
        return True

    # ---------------------------------------------------------- page state
    def stats(self):
        with self.lock:
            up = int(time.time() - self.started_at) if (self.running and self.started_at) else 0
            e = self.enroll
            return {
                "running": self.running, "source": self.source, "count": self.count, "peak": self.peak,
                "avg_conf": round(self.conf_sum / self.conf_n * 100, 1) if self.conf_n else 0,
                "logged": self.detections_logged, "uptime": up, "fps": round(self.fps, 1), "ms": round(self.ms),
                "faces": self.faces, "alert": bool(self.settings["alerts"] and self.count > self.settings["capacity"]),
                "settings": self.settings, "history": self.history[-60:],
                "recognizer": bool(self.rec), "known_people": [c for c in (self.rec.classes if self.rec else []) if c != fr.UNKNOWN],
                "enroll": None if not e else {"name": e["name"], "saved": e["saved"], "target": e["target"],
                                               "msg": e["msg"], "done": e["done"]},
                "train": self.train_state,
            }


# ====================================================================== web server
ENGINE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # keep the terminal quiet
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def jbody(self):
        try:
            return json.loads(self.body() or b"{}")
        except Exception:
            return {}

    # ---------------------------------------------------------- GET
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/video":
            return self.stream()
        if path == "/api/stats":
            return self.send(200, ENGINE.stats())
        if path == "/api/people":
            return self.send(200, ENGINE.people())
        if path == "/api/snapshot.jpg":
            if not ENGINE.jpeg:
                return self.send(404, {"error": "camera is off"})
            OUT.mkdir(exist_ok=True); (OUT / "screenshots").mkdir(exist_ok=True)
            (OUT / "screenshots" / f"web_{datetime.now():%Y%m%d_%H%M%S}.jpg").write_bytes(ENGINE.jpeg)
            return self.send(200, ENGINE.jpeg, "image/jpeg",
                             {"Content-Disposition": f'attachment; filename="crowdvision_{datetime.now():%H%M%S}.jpg"'})
        if path == "/api/history.csv":
            s = io.StringIO(); w = csv.writer(s); w.writerow(["time", "source", "faces"])
            for h in ENGINE.history:
                w.writerow([h["time"], h["source"], h["count"]])
            return self.send(200, s.getvalue(), "text/csv",
                             {"Content-Disposition": 'attachment; filename="crowdvision_history.csv"'})
        if path == "/api/registry.csv":
            s = io.StringIO(); w = csv.writer(s)
            w.writerow(["name", "tag", "category", "photos", "trained", "times_seen", "first_seen", "last_seen", "notes"])
            for p in ENGINE.people():
                w.writerow([p["name"], p["tag"], p["category"], p["photos"], p["trained"], p["times_seen"],
                            p["first_seen"], p["last_seen"], p["notes"]])
            return self.send(200, s.getvalue(), "text/csv",
                             {"Content-Disposition": 'attachment; filename="crowdvision_registry.csv"'})
        return self.send(404, {"error": "not found"})

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                jpg = ENGINE.jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.03)
                if not ENGINE.running and jpg is None:
                    time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # ---------------------------------------------------------- POST
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/start":
            ok, msg = ENGINE.start(self.jbody().get("source", "0"))
            return self.send(200 if ok else 400, {"ok": ok, "message": msg})
        if path == "/api/stop":
            ENGINE.stop()
            return self.send(200, {"ok": True})
        if path == "/api/settings":
            b = self.jbody()
            for k in ("threshold", "capacity", "alerts", "logging", "show_conf", "recognition"):
                if k in b:
                    ENGINE.settings[k] = b[k]
            return self.send(200, ENGINE.settings)
        if path == "/api/analyze":
            res = ENGINE.analyze_image(self.body())
            if res is None:
                return self.send(400, {"error": "could not read that image"})
            jpg, faces = res
            import base64
            return self.send(200, {"image": base64.b64encode(jpg).decode(), "faces": faces, "count": len(faces)})
        if path == "/api/video":
            q = parse_qs(urlparse(self.path).query)
            name = Path(q.get("name", ["upload.mp4"])[0]).name
            (OUT / "uploads").mkdir(parents=True, exist_ok=True)
            p = OUT / "uploads" / name
            p.write_bytes(self.body())
            ENGINE.stop()
            ok, msg = ENGINE.start(str(p))
            return self.send(200 if ok else 400, {"ok": ok, "message": msg})
        if path == "/api/enroll":
            b = self.jbody()
            ok, msg = ENGINE.start_enroll(b.get("name", ""), b.get("tag", ""), b.get("category", ""), b.get("notes", ""))
            return self.send(200 if ok else 400, {"ok": ok, "message": msg})
        if path == "/api/enroll/stop":
            ENGINE.enroll = None
            return self.send(200, {"ok": True})
        if path == "/api/train":
            started = ENGINE.start_training()
            return self.send(200, {"ok": started})
        if path == "/api/delete":
            ENGINE.delete_person(self.jbody().get("name", ""))
            return self.send(200, {"ok": True})
        return self.send(404, {"error": "not found"})


def main():
    global ENGINE
    ap = argparse.ArgumentParser(description="CrowdVision web console")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=640, help="processing width (480 = faster)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    import os
    os.chdir(ROOT)                      # data/, outputs/ and models are relative to the project folder
    print("Loading CrowdVisionNet ...")
    ENGINE = Engine(width=args.width)
    print("Recognition model:", "loaded" if ENGINE.rec else "not trained yet (enroll people + train from the Registry)")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nCrowdVision console running at {url}   (press Ctrl+C here to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
