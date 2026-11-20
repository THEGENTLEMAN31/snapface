"""Tests des effets : forme, bornes, identité, ancrage sur landmarks."""
import numpy as np
import pytest
import torch

from snapface.effects import available_effects, get_effect
from snapface.effects.base import create_grid, sample

SIZE = 64
ANCHORS = {"eye_l": (0.30, 0.40), "eye_r": (0.70, 0.40), "nose": (0.50, 0.55),
           "mouth_l": (0.42, 0.72), "mouth_r": (0.58, 0.72)}


def _crop(h=SIZE, w=SIZE):
    torch.manual_seed(0)
    return torch.rand(1, 3, h, w)


@pytest.mark.parametrize("name", available_effects())
def test_shape_and_range(name):
    effect = get_effect(name, device="cpu")
    crop = _crop()
    out = effect.apply(crop, ANCHORS)
    assert out.shape == crop.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_identity_is_identity():
    effect = get_effect("identity", device="cpu")
    crop = _crop()
    out = effect.apply(crop)
    assert (out - crop).abs().max().item() < 1e-4


@pytest.mark.parametrize("name", ["bulge", "pinch", "swirl", "mirror", "kaleidoscope"])
def test_non_identity_changes_image(name):
    effect = get_effect(name, device="cpu")
    crop = _crop()
    assert (effect.apply(crop, ANCHORS) - crop).abs().max().item() > 1e-3


def test_randoms_do_not_break_bounds():
    crop = _crop()
    for name in available_effects():
        for seed in range(3):
            torch.manual_seed(seed)
            effect = get_effect(name, device="cpu")
            out = sample(crop, effect.grid_for(SIZE, SIZE, ANCHORS))
            assert float(out.min()) >= -1e-6 and float(out.max()) <= 1.0 + 1e-6


def test_big_eyes_is_anchored_on_eye():
    """Le déplacement maximal doit se trouver près de l'oeil, pas au centre."""
    effect = get_effect("big_eyes", device="cpu")
    grid = effect.grid_for(SIZE, SIZE, ANCHORS)
    base = create_grid(SIZE, SIZE, "cpu")
    du = (grid - base)[0, 0].abs() + (grid - base)[0, 1].abs()
    iy, ix = np.unravel_index(int(du.argmax()), du.shape)
    u = ix / (SIZE - 1) * 2.0 - 1.0
    v = iy / (SIZE - 1) * 2.0 - 1.0
    eye_u = 2.0 * ANCHORS["eye_l"][0] - 1.0
    eye_v = 2.0 * ANCHORS["eye_l"][1] - 1.0
    assert abs(u - eye_u) < 0.2 and abs(v - eye_v) < 0.2


def test_gain_modulates_amplitude():
    weak = get_effect("bulge", device="cpu", params={"strength": 0.1})
    strong = get_effect("bulge", device="cpu", params={"strength": 0.6})
    crop = _crop()
    a = (weak.apply(crop) - crop).abs().max().item()
    b = (strong.apply(crop) - crop).abs().max().item()
    assert b > a


def test_morph_between_effects_is_continuous():
    from snapface.effects.base import morph_grid
    a = get_effect("bulge", device="cpu").grid_for(SIZE, SIZE, ANCHORS)
    b = get_effect("swirl", device="cpu").grid_for(SIZE, SIZE, ANCHORS)
    mid = morph_grid(a, b, 0.5)
    assert (mid - a).abs().max() > 0
    assert torch.allclose(morph_grid(a, b, 0.0), a)
    assert torch.allclose(morph_grid(a, b, 1.0), b)
