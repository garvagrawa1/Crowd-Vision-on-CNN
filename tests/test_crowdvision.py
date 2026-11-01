"""Automated tests for CrowdVision (run:  python -m pytest -v tests)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import CrowdVisionNet, count_parameters                       # noqa: E402
from detector import box_iou_matrix, nms, cluster_merge, load_detector   # noqa: E402
from detect import FaceTracker, crowd_level                              # noqa: E402
from prepare_data import split_of, iou                                   # noqa: E402
import face_recognition_app as fr                                        # noqa: E402

LFW = ROOT / "datasets" / "lfw" / "lfw-deepfunneled"


# ---------------------------------------------------------------- unit tests: model
def test_ut01_crop_output_shape():
    net = CrowdVisionNet().eval()
    c, b = net(torch.zeros(1, 3, 48, 48))
    assert tuple(c.shape) == (1, 2, 1, 1) and tuple(b.shape) == (1, 4, 1, 1)


def test_ut02_fully_convolutional_frame_output():
    net = CrowdVisionNet().eval()
    c, _ = net(torch.zeros(1, 3, 480, 640))
    assert tuple(c.shape) == (1, 2, 55, 75)


def test_ut03_parameter_count():
    assert count_parameters(CrowdVisionNet()) == 122774


def test_ut04_full_frame_equals_sliding_window():
    torch.manual_seed(0)
    net = CrowdVisionNet().eval()
    img = torch.rand(1, 3, 96, 96)
    full, _ = net(img)
    crop, _ = net(img[:, :, 8:56, 16:64])          # window at row 1, column 2 (stride 8)
    assert torch.allclose(full[0, :, 1, 2], crop[0, :, 0, 0], atol=1e-5)


# ---------------------------------------------------------------- unit tests: geometry and NMS
def test_ut05_iou_values():
    a = np.array([[0, 0, 10, 10]], float)
    iou_same, _ = box_iou_matrix(a, a)
    iou_far, _ = box_iou_matrix(a, np.array([[20, 20, 30, 30]], float))
    assert abs(iou_same[0, 0] - 1) < 1e-6 and iou_far[0, 0] == 0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-6


def test_ut06_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_ut07_cluster_merge_votes():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]], float)
    out, scores, votes = cluster_merge(boxes, np.array([0.99, 0.98, 0.99]), min_votes=2)
    assert len(out) == 1 and votes[0] == 2           # lone box dropped, pair merged


def test_ut08_nested_box_rule():
    boxes = np.array([[40, 40, 60, 60], [41, 41, 61, 61], [0, 0, 100, 100], [1, 1, 101, 101]], float)
    out, _, _ = cluster_merge(boxes, np.array([0.995, 0.995, 0.999, 0.999]), min_votes=2)
    assert len(out) == 1 and out[0][2] - out[0][0] < 30   # only the small (real) face box remains


# ---------------------------------------------------------------- unit tests: tracking, alerts, data
def test_ut09_tracker_needs_two_hits():
    t = FaceTracker(min_hits=2)
    assert t.update([(10, 10, 50, 50, 0.99)]) == []
    assert len(t.update([(11, 10, 51, 50, 0.99)])) == 1


def test_ut10_tracker_keeps_id():
    t = FaceTracker(min_hits=1)
    first = t.update([(10, 10, 50, 50, 0.99)])[0].id
    second = t.update([(14, 12, 54, 52, 0.99)])[0].id
    assert first == second


def test_ut11_crowd_levels():
    assert crowd_level(1, 10)[0] == "LOW"
    assert crowd_level(8, 10)[0] == "HIGH"
    assert crowd_level(11, 10)[0] == "OVER CAPACITY"


def test_ut12_person_split_is_deterministic():
    assert split_of("Colin_Powell") == split_of("Colin_Powell")
    assert {split_of(f"person_{i}") for i in range(200)} == {0, 1, 2}


def test_ut13_face_crop_at_frame_edge():
    frame = np.zeros((100, 100, 3), np.uint8)
    crop = fr.face_crop(frame, (-20, -20, 40, 40))
    assert crop.shape == (64, 64, 3)


# ---------------------------------------------------------------- integration tests
@pytest.fixture(scope="module")
def det():
    return load_detector(str(ROOT / "crowdvision_cnn.pth"), threshold=0.99, min_face=48)


def test_it01_blank_frame_has_no_faces(det):
    assert det.detect(np.full((480, 640, 3), 127, np.uint8)) == []


@pytest.mark.skipif(not LFW.exists(), reason="LFW dataset not downloaded")
def test_it02_detects_face_in_unseen_photo(det):
    import cv2
    people = sorted(p for p in LFW.iterdir() if split_of(p.name) == 2)
    img = cv2.imread(str(next(people[0].glob("*.jpg"))))
    found = det.detect(img)
    assert len(found) == 1
    assert iou(found[0][:4], (75, 80, 175, 180)) >= 0.4


def test_it03_trained_model_loads_and_runs(det):
    assert det.net.training is False and len(det.scales_for(480, 640)) >= 3


@pytest.mark.skipif(not (ROOT / fr.RECOG_PATH).exists(), reason="recognizer not trained")
def test_it04_recognizer_returns_valid_names():
    rec = fr.Recognizer(ROOT / fr.RECOG_PATH)
    out = rec.predict([np.zeros((64, 64, 3), np.uint8)])
    assert out[0][0] in rec.classes and 0 <= out[0][1] <= 1
