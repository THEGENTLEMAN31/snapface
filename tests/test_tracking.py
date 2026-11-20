"""Tests du suivi temporel : One Euro et propagation par flux optique."""
import cv2
import numpy as np

from snapface.detector import Detection
from snapface.tracking import BoxTracker, OneEuro, anchors_dict


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(1)
    noisy = 100.0 + rng.normal(0, 2.0, 120)
    filt = OneEuro(min_cutoff=1.0, beta=0.01)
    out = [filt(x, i / 30.0) for i, x in enumerate(noisy)]
    assert np.std(out[10:]) < np.std(noisy[10:])


def test_one_euro_tracks_a_ramp():
    filt = OneEuro(min_cutoff=2.0, beta=0.02)
    vals = [filt(x, i / 30.0) for i, x in enumerate(np.linspace(0, 100, 90))]
    assert abs(vals[-1] - 100.0) < 15.0


def _scene(dx: int = 0, dy: int = 0) -> np.ndarray:
    """Texture aléatoire lissée : des gradients partout, pour que LK accroche."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def _det() -> Detection:
    pts = [(20.0, 15.0), (40.0, 15.0), (30.0, 25.0), (23.0, 33.0), (37.0, 33.0)]
    return Detection(box=np.array([10.0, 5.0, 50.0, 50.0], float),
                     score=0.9, landmarks=pts)


def test_tracker_seeds_on_detection():
    tr = BoxTracker()
    box = tr.update(_det(), _scene(), 0.0)
    assert box is not None
    assert abs(box[2] - 50.0) < 1.0 and abs(box[3] - 50.0) < 1.0
    a = anchors_dict(tr.anchors)
    assert a is not None and 0.0 < a["nose"][0] < 1.0


def test_tracker_follows_translation_without_detection():
    tr = BoxTracker()
    tr.update(_det(), _scene(0, 0), 0.0)
    moved = None
    for i in range(1, 6):
        moved = tr.update(None, _scene(6 * i, 3 * i), i / 30.0)
    assert moved is not None
    assert moved[0] > 10.0 + 4.0, "la bbox doit suivre le visage vers la droite"
    assert tr.stats["tracked"] > 0


def test_tracker_expires_when_face_disappears():
    tr = BoxTracker(expire=0.2)
    tr.update(_det(), _scene(), 0.0)
    assert tr.update(None, _scene(), 0.5) is None


def test_tracker_reports_lost():
    tr = BoxTracker(expire=0.05)
    tr.update(_det(), _scene(), 0.0)
    tr.update(None, _scene(400, 400), 1.0)
    assert tr.stats["lost"] >= 1
