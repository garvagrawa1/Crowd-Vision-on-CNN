# CrowdVision — Face Detection & Crowd Counting with Our Own CNN

Full project report (PDF): [`docs/report/CrowdVision_SE_Mini_Project_Report.pdf`](docs/report/CrowdVision_SE_Mini_Project_Report.pdf)

CrowdVision detects faces in a live webcam feed, video or photo and counts the people
it can see. The detector is **our own convolutional neural network, `CrowdVisionNet`,
trained completely from scratch**.

- No YOLO and no pretrained weights.
- No OpenCV / dlib / MediaPipe face detector.
- OpenCV is only used to read the camera and draw on the screen.
- PyTorch only provides the basic layers.

---

## 1. Quick start (VS Code terminal, Windows)

Open this folder in VS Code, then open the terminal (`Ctrl + ~`).

```powershell
# one-time setup
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# run the live demo with the already-trained model
python detect.py
```

If PowerShell refuses to activate the venv, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Demo commands

| Command | What it does |
|---|---|
| `python detect.py` | Laptop webcam |
| `python detect.py --source 1` | External / USB webcam |
| `python detect.py --source video.mp4` | Video file |
| `python detect.py --source photo.jpg` | Single image; the result is saved in `outputs/` |
| `python detect.py --capacity 3` | Alert when more than 3 people are visible |
| `python detect.py --width 480` | Faster on slow laptops |
| `python detect.py --threshold 0.995` | Even fewer false boxes, may miss some faces (default 0.99) |
| `python detect.py --threshold 0.97` | Finds harder faces, but more false boxes |
| `python detect.py --min-face 30` | Finds smaller, more distant faces (slower) |

### Keys while it runs

- **q / Esc**: quit
- **p / Space**: pause
- **s**: screenshot
- **r**: record video
- **+ / −**: change the confidence threshold
- **Mouse drag**: draw a counting zone; right-click clears it

Counts are logged once per second to `outputs/logs/*.csv`.

---

## 2. Re-training the model yourself (full pipeline)

```powershell
python download_datasets.py   # LFW faces + Caltech-101 scenes (~260 MB, unpacks them)
python prepare_data.py        # builds ~166k 48x48 training crops (~1 min)
python train.py               # trains from scratch + hard-negative mining (30-60 min on CPU)
python evaluate.py            # precision / recall / counting accuracy on unseen scenes
python detect.py              # live demo with your new model
```

`python train.py --epochs 3 --finetune-epochs 1` is a quick test run.

**Fewer false boxes in your demo room:**
1. Run `python collect_backgrounds.py` with **nobody** in front of the camera.
2. Press Space to save 15–30 frames.
3. Run `prepare_data.py` and then `train.py` again. The model learns that your walls,
   curtains and furniture are not faces.

---

## 3. How it works

```
webcam frame ──► image pyramid ──► CrowdVisionNet (fully convolutional, 1 pass per level)
                                        │
                     face-probability map + box offsets (one per 48x48 window, every 8 px)
                                        │
          threshold ──► box refinement ──► clustered NMS with voting ──► IoU tracker ──► count
```

### CrowdVisionNet (`model.py`, 122,774 parameters)

| Layer | Output for a 48×48 crop |
|---|---|
| Conv 3×3, 16 filters + BatchNorm + ReLU, MaxPool 2 | 23×23×16 |
| Conv 3×3, 32 filters + BatchNorm + ReLU, MaxPool 2 | 10×10×32 |
| Conv 3×3, 64 filters + BatchNorm + ReLU, MaxPool 2 | 4×4×64 |
| Conv 4×4, 96 filters (acts as the fully connected layer) + Dropout | 1×1×96 |
| Head 1: Conv 1×1 → 2 | face / non-face scores |
| Head 2: Conv 1×1 → 4 | box offsets (x1, y1, x2, y2) |

**Why fully convolutional?** The network has no `Linear` layers, so it can take a whole frame
at once. It then returns a prediction for every 48×48 window (8 px apart) in one forward
pass. This gives exactly the same answers as the old sliding-window loop, but runs about
50× faster, because neighbouring windows share their computation. All convolutions are
unpadded, so the results on full frames match training exactly.

### Training data (`prepare_data.py`)

| Crop type | Source | Rule | Label |
|---|---|---|---|
| Positive | LFW face photos | IoU ≥ 0.65 with the face | face + box offsets |
| Part face | LFW | 0.4 ≤ IoU < 0.65 | box offsets only (teaches precise boxes) |
| Negative | LFW (hair, shoulders, half faces) + Caltech-101 scenes and objects | IoU < 0.3 | non-face |

People are split **by identity** into train (80%), validation (10%) and test (10%). The
model is never tested on a person it saw during training.

### Training (`train.py`)

- **Training setup:** AdamW optimiser, cosine learning-rate schedule.
- **Loss:** cross-entropy with online hard-example mining (learns from the hardest 70% of
  each batch), plus Smooth-L1 for the box offsets.
- **Augmentation:**
  - brightness and contrast
  - white balance
  - greyscale
  - low resolution
  - blur
  - sensor noise
- **Hard-negative mining (bootstrapping):** after the first training stage, the detector is
  run on thousands of face-free photos. Every false alarm is added back as a "non-face"
  example and the model is fine-tuned. This is the main reason it makes few false boxes.

### Detection (`detector.py`) — how duplicate boxes are fixed

1. Windows above the threshold are refined with the predicted box offsets. Windows lying
   mostly in the black padding outside the frame are dropped.
2. **Clustered NMS with voting:** overlapping boxes *of similar size* are merged into one
   confidence-weighted average box. A box must be supported by at least 2 windows, so
   one-off texture hits are dropped.
3. **Nested-box rule:** one face cannot contain another face. A big box around a smaller
   detected face (face + surroundings) is removed, and of two same-size duplicates the more
   confident one is kept.
3. **Tracker** (`detect.py`): each face gets an ID and its box is smoothed over time. A box
   is shown only after 2 consecutive hits, and the count is the median of recent frames.
   The count therefore stays steady instead of flickering.

---

## 4. Results

**Crop-level test** (3,065 face and 11,762 non-face crops of people never seen in training):
accuracy 99.87%, precision 99.87%, recall 99.48%, F1 99.67%.

**Full detection on 300 unseen crowd scenes** (640×480, 1–10 faces each, threshold 0.99, IoU ≥ 0.4):

| Metric | Value |
|---|---|
| Precision | 96.1% |
| Recall | 99.0% |
| F1 score | 97.5% |
| Count error (MAE) | 0.23 people per image |
| Exact count | 82.3% of images |

Training: 10 epochs + hard-negative mining (40,000 mined false alarms) + 4 fine-tune epochs,
about 30 minutes on a 2-core CPU.


See `results/` after training and evaluation:

- `training_curves.png`: loss, accuracy, precision and recall per epoch
- `test_metrics.txt`: crop-level test accuracy and confusion matrix (people never seen in training)
- `eval_metrics.txt` + `pr_curve.png`: detection precision, recall, F1 and counting error on
  300 unseen multi-face scenes
- `eval_examples/`: example scenes (blue = ground truth, green = prediction)

---

## Face Recognition (who is in the crowd)

`face_recognition_app.py` adds names to the CrowdVision screen using **two of our own CNNs**:
- `crowdvision_cnn.pth` finds the faces.
- `recognizer_cnn.pth` (RecogNet, trained from scratch on your enrolled people) says who each face is, or "Unknown".

```powershell
python face_recognition_app.py
```

The menu offers these options:
1. **Enroll a person.** Type the name. The webcam captures about 60 photos automatically; move your head slowly. Only that person should be in view.
2. **Train recognizer.** Takes about 1–2 minutes on CPU and prints the accuracy for each person.
3. **Start live recognition.** Shows a name and confidence on each green box, red **Unknown** boxes for strangers,
   the known/unknown count, the crowd alert and FPS. **q** quits, **s** saves a screenshot. An attendance file
   (name, first seen, last seen) is saved in `outputs/logs/` when you quit.

Enroll at least 2 people (more photos means better accuracy), then train, then run. To add someone
later, enroll them and train again.

Command-line form: `python face_recognition_app.py enroll --name Rahul`, `... train`, `... run --capacity 5`.

## 5. Automated tests

```powershell
python -m pytest tests -v   # 17 tests
```

## 6. Limitations

- It counts **visible faces**. People turned away, hidden or very small (< ~30 px) are not counted.
- It is trained mostly on frontal and near-frontal faces (LFW), so strong side profiles can be missed.
- The evaluation scenes are synthetic composites. Real crowd footage is harder.
- It runs on a CPU. Speed depends on `--width` and `--min-face`.

## 7. Files

| File | Purpose |
|---|---|
| `model.py` | CrowdVisionNet architecture |
| `download_datasets.py` | Downloads LFW + Caltech-101 |
| `prepare_data.py` | Builds 48×48 positive / part / negative crops |
| `train.py` | Training + hard-negative mining + metrics and plots |
| `detector.py` | Pyramid inference, box refinement, clustered NMS |
| `detect.py` | Real-time app (webcam / video / image, tracker, zone, alerts, logging) |
| `evaluate.py` | Precision / recall / counting evaluation on unseen scenes |
| `collect_backgrounds.py` | Captures room photos for extra hard negatives |
| `crowdvision_cnn.pth` | Trained weights |
| `web_app.py` | Web console (live feed, photo analyser, recognition, dashboard) |
| `face_recognition_app.py` | Enrol people, train RecogNet, live recognition and attendance |
| `web/index.html` | Web console front end |
| `tests/test_crowdvision.py` | 17 automated unit and integration tests |
| `results/` | Training log, metrics, plots and example evaluation scenes |
| `docs/` | Project report (PDF and Word) and screenshots |
