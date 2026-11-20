"""Test d'intégration : le pipeline complet sur une image, sans display."""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from snapface.config import Config
from snapface.pipeline import Pipeline

TMP = Path("/tmp/opencode/sftest")


@pytest.fixture(scope="module")
def photo() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / "lena.jpg"
    if not dst.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            pytest.skip("photo de test indisponible")
    return dst


@pytest.mark.parametrize("effect", ["identity", "bulge", "kaleidoscope"])
def test_image_mode_writes_clean_file(photo: Path, effect: str, tmp_path: Path):
    out = tmp_path / f"{effect}.png"
    cfg = Config(effect=effect, clean=True, record=out, device="cpu")
    code = Pipeline(cfg, source="image", image_path=str(photo)).run()
    assert code == 0
    img = cv2.imread(str(out))
    assert img is not None and img.size > 0


def test_image_mode_actually_changes_pixels(photo: Path, tmp_path: Path):
    src = cv2.imread(str(photo))
    outs = {}
    for effect in ("identity", "swirl"):
        dst = tmp_path / f"{effect}.png"
        Pipeline(Config(effect=effect, clean=True, record=dst, device="cpu"),
                 source="image", image_path=str(photo)).run()
        outs[effect] = cv2.imread(str(dst))
    diff = np.abs(outs["swirl"].astype(int) - outs["identity"].astype(int))
    assert diff.max() > 10, "l'effet swirl doit modifier l'image"
    assert (diff > 10).mean() < 0.35, "la modification reste localisée au visage"


def test_unknown_effect_raises():
    with pytest.raises(ValueError):
        Pipeline(Config(effect="nope"), source="image", image_path="x.jpg")
