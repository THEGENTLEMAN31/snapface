"""Rend des PNG avant/après pour chaque effet, sans webcam, avec assertions."""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from snapface.detector import FaceDetector
from snapface.effects import available_effects, get_effect
from snapface.effects.base import sample
from snapface.geometry import (anchors_in_crop, crop_face, from_tensor,
                               landmarks_to_norm, paste_face, seam_error, to_tensor)

TMP = Path("/tmp/opencode/sftest")
DEFAULT_IMAGE = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                 "samples/data/lena.jpg")


def _load_source(image_path: str | None, want: str = "face") -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is not None:
            return img
        print(f"Impossible de lire {image_path}")
    TMP.mkdir(parents=True, exist_ok=True)
    for name in ("lena.jpg", "messi5.jpg"):
        dst = TMP / name
        if not dst.exists():
            try:
                urllib.request.urlretrieve(
                    f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{name}",
                    dst)
            except Exception:
                continue
        if dst.exists() and dst.stat().st_size > 1000:
            return cv2.imread(str(dst))
    return _synthetic_face()


def _synthetic_face(size: int = 512) -> np.ndarray:
    img = np.full((size, size, 3), 210, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 + 20), 0, 0, 360,
                (200, 180, 160), -1)
    cv2.circle(img, (size // 2 - size // 8, size // 2 - size // 10), 14, (30, 30, 30), -1)
    cv2.circle(img, (size // 2 + size // 8, size // 2 - size // 10), 14, (30, 30, 30), -1)
    cv2.ellipse(img, (size // 2, size // 2 + size // 8), (size // 9, size // 7), 0, 0, 180,
                (30, 30, 30), 3)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Échantillons PNG par effet + index.html.")
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--image", default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--conf", type=float, default=0.6)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda demandé mais CUDA indisponible, CPU utilisé.")
        device = "cpu"

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_source(args.image)
    detector = FaceDetector(conf_threshold=args.conf)
    det = detector.detect(frame)
    print(f"Détecteur: {detector.using}" + ("" if det else " (aucun visage détecté)"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    links, rows = [], []

    crop, cbox = crop_face(frame, det.box if det else (0, 0, 256, 256))
    anchors = anchors_in_crop(
        landmarks_to_norm(det.landmarks, det.box) if det else None,
        det.box if det else (0, 0, 256, 256), cbox)
    crop_sq = cv2.resize(crop, (256, 256))
    t_in = to_tensor(crop_sq, device)

    for name in available_effects():
        effect = get_effect(name, device=device)
        t_out = effect.apply(t_in, anchors)
        assert t_out.shape == t_in.shape, f"{name}: forme modifiée"
        assert float(t_out.min()) >= 0.0 and float(t_out.max()) <= 1.0, f"{name}: hors bornes"
        if name == "identity":
            err = (t_out - t_in).abs().max().item()
            assert err < 1e-4, f"identity err {err}"
        face = cv2.resize(from_tensor(t_out), (int(cbox[2]), int(cbox[3])))
        rendered = paste_face(frame.copy(), face, cbox, feather=15)
        seam = seam_error(frame, rendered, cbox)
        path = out_dir / f"{name}_{stamp}.png"
        cv2.imwrite(str(path), rendered)
        links.append(f'<img src="{path.name}" alt="{name}" width="240"/>')
        rows.append(f"[OK] {name:14s} couture_max={seam:3.0f} -> {path}")
        print(rows[-1])

    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>SnapFace samples</title></head><body>"
            f"<h1>SnapFace - échantillons ({stamp}, {device})</h1>"
            + "".join(f"<div style='display:inline-block;margin:4px'><p>{n}</p>{l}</div>"
                      for n, l in zip(available_effects(), links))
            + "</body></html>")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[OK] index.html ({len(links)} effets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
