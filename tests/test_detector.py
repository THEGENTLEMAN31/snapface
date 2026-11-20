"""Tests du détecteur : format de sortie OpenCV 5 et cohérence des landmarks."""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from snapface.detector import FaceDetector

TMP = Path("/tmp/opencode/sftest")


def _image(name: str = "messi5.jpg") -> np.ndarray:
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / name
    if not dst.exists():
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{name}"
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            pytest.skip("image de test indisponible")
    img = cv2.imread(str(dst))
    if img is None:
        pytest.skip("image illisible")
    return img


def test_detector_available():
    det = FaceDetector(0.6)
    assert det.using == "yunet"


def test_score_is_a_probability():
    """Régression : le score est la DERNIÈRE colonne, pas l'index 4."""
    img = _image()
    det = FaceDetector(0.6).detect(img)
    assert det is not None
    assert 0.0 < det.score <= 1.0, f"score hors [0,1] : {det.score}"


def test_landmarks_are_inside_the_box():
    img = _image()
    det = FaceDetector(0.6).detect(img)
    x, y, w, h = det.box
    assert det.landmarks is not None and len(det.landmarks) == 5
    for px, py in det.landmarks:
        assert x - 0.1 * w <= px <= x + 1.1 * w
        assert y - 0.1 * h <= py <= y + 1.1 * h


def test_eyes_above_nose_above_mouth():
    """Ordre anatomique : yeux, nez, coins de bouche."""
    img = _image()
    det = FaceDetector(0.6).detect(img)
    pts = np.array(det.landmarks, float)
    eyes_y = pts[:2, 1].mean()
    nose_y = pts[2, 1]
    mouth_y = pts[3:, 1].mean()
    assert eyes_y < nose_y < mouth_y, f"landmarks incohérents : {pts.round(1)}"
