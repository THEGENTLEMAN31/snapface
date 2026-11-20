"""Tests de géométrie : masque de fusion, crop carré, absence de couture."""
import cv2
import numpy as np

from snapface.geometry import (anchors_in_crop, crop_face, feather_mask,
                               paste_face, seam_error, square_box)


def _frame(h=240, w=320):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_feather_mask_shape_values():
    m = feather_mask(80, 100, 12)
    assert m.shape == (80, 100)
    assert m[40, 50] == 1.0
    assert m[0, 0] == 0.0 and m[0, 50] == 0.0 and m[40, 0] == 0.0
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_feather_mask_is_monotonic_towards_center():
    m = feather_mask(64, 64, 10)
    row = m[32, :32]
    assert np.all(np.diff(row) >= -1e-6)


def test_square_box_is_square_and_inside():
    box = (100.0, 80.0, 100.0, 60.0)
    bx, by, bw, bh = square_box(box, (240, 320), expand=0.35)
    assert abs(bw - bh) < 1e-6
    assert bx >= 0 and by >= 0 and bx + bw <= 320 and by + bh <= 240
    # le centre est conservé quand la boîte carrée tient dans le frame
    assert abs((bx + bw / 2) - 150.0) < 1.0
    assert abs((by + bh / 2) - 110.0) < 1.0


def test_square_box_is_shifted_not_squeezed_near_border():
    box = (2.0, 2.0, 60.0, 40.0)
    bx, by, bw, bh = square_box(box, (240, 320), expand=0.35)
    assert bx == 0.0 and by == 0.0, "la boîte doit être rabattue, pas rétrécie"
    assert abs(bw - bh) < 1e-6


def test_crop_face_returns_square_crop():
    frame = _frame()
    crop, cbox = crop_face(frame, (100.0, 70.0, 90.0, 70.0))
    assert crop.shape[0] == crop.shape[1]
    assert cbox[2] == cbox[3]


def test_paste_identity_has_no_seam():
    """Recoller le crop tel quel ne doit produire aucune couture."""
    frame = _frame()
    crop, cbox = crop_face(frame, (100.0, 70.0, 90.0, 70.0))
    out = paste_face(frame.copy(), crop, cbox, feather=12)
    assert seam_error(frame, out, cbox) <= 2
    assert np.abs(out.astype(int) - frame.astype(int)).max() <= 2


def test_paste_feathered_edge_is_progressive():
    """Sur un patch uni, le bord doit être un dégradé, pas une marche."""
    frame = np.zeros((120, 120, 3), np.uint8)
    patch = np.full((60, 60, 3), 255, np.uint8)
    out = paste_face(frame.copy(), patch, (30, 30, 60, 60), feather=10)
    column = out[30:90, 60, 0].astype(int)  # coupe verticale au centre du patch
    assert column[0] < 60, "le bord doit être atténué"
    assert column[30] > 200, "le centre doit être plein"
    assert np.abs(np.diff(column)).max() <= 40, "pas de marche brutale"


def test_anchors_in_crop_are_normalised():
    anchors = np.array([[0.3, 0.4], [0.7, 0.4], [0.5, 0.55], [0.42, 0.72], [0.58, 0.72]],
                       np.float32)
    box = (100.0, 50.0, 100.0, 100.0)
    crop_box = (75.0, 25.0, 150.0, 150.0)
    a = anchors_in_crop(anchors, box, crop_box)
    assert set(a) == {"eye_r", "eye_l", "nose", "mouth_r", "mouth_l"}
    for u, v in a.values():
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
