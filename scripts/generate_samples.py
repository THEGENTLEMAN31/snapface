"""Génère des échantillons PNG avant/après pour chaque effet, sans webcam."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from snapface.config import YUNET_PATH
from snapface.detector import FaceDetector
from snapface.effects import get_effect, available_effects
from snapface.geometry import to_tensor, from_tensor, crop_face, paste_face

_DEFAULT_IMAGE = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"


def _download_image(url: str, dest: Path) -> Path:
    import urllib.request
    print(f"Téléchargement de l'image de test: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _synthetic_face(size: int = 512) -> np.ndarray:
    img = np.full((size, size, 3), 210, dtype=np.uint8)
    img = cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 + 20), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(img, (size // 2 - size // 8, size // 2 - size // 10), 14, (30, 30, 30), -1)
    cv2.circle(img, (size // 2 + size // 8, size // 2 - size // 10), 14, (30, 30, 30), -1)
    cv2.ellipse(img, (size // 2, size // 2 + size // 8), (size // 9, size // 7), 0, 0, 180, (30, 30, 30), 3)
    return img


def _load_source(image_path: str | None, tmp: Path) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is not None:
            return img
        print(f"Impossible de lire {image_path}, image synthétique utilisée.")
    for name in ("lena.jpg", "messi5.jpg"):
        dest = tmp / name
        if not dest.exists():
            try:
                _download_image(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{name}", dest)
            except Exception:
                continue
        if dest.exists() and dest.stat().st_size > 1000:
            return cv2.imread(str(dest))
    return _synthetic_face()


def main() -> int:
    ap = argparse.ArgumentParser(description="Rend des PNG avant/après pour chaque effet SnapFace.")
    ap.add_argument("--dir", type=str, default="samples", help="Répertoire de sortie (défaut: samples)")
    ap.add_argument("--image", type=str, default=None, help="Image source (photo de face)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--conf", type=float, default=0.6)
    args = ap.parse_args()

    import torch as _t
    if args.device == "cuda" and not _t.cuda.is_available():
        print("[WARN] --device cuda mais torch.cuda.is_available()=False, CPU utilisé.")
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if _t.cuda.is_available() else "cpu"

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/opencode/test")
    tmp.mkdir(parents=True, exist_ok=True)
    frame = _load_source(args.image, tmp)

    detector = FaceDetector(conf_threshold=args.conf)
    print(f"Détecteur: {detector.using}")
    det = detector.detect(frame)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    links, rows = [], []
    for name in available_effects():
        effect = get_effect(name, device=device)
        if det is None:
            base = cv2.resize(frame, (256, 256))
            crop, box = base, (0, 0, base.shape[1], base.shape[0])
        else:
            crop, box = crop_face(frame, det, expand=0.35)
            crop = cv2.resize(crop, (256, 256))
        t_in = to_tensor(crop, device)
        t_out = effect.apply(t_in)
        assert t_out.shape == t_in.shape, "forme modifiée par l'effet"
        assert float(t_out.min()) >= 0.0 and float(t_out.max()) <= 1.0, "bornes hors domaine"
        if name == "identity":
            err = (t_out - t_in).abs().max().item()
            assert err < 1e-3, f"identity err {err}"
        out_img = from_tensor(t_out)
        if det is not None:
            out_img = cv2.resize(out_img, (box[2], box[3]))
            rendered = paste_face(frame.copy(), out_img, box, feather=15)
        else:
            rendered = out_img
        base_name = f"{name}_{stamp}.png"
        path = out_dir / base_name
        cv2.imwrite(str(path), rendered)
        links.append(f'<img src="{base_name}" alt="{name}" width="240"/>')
        rows.append(f"[OK] {name:14s} -> {path}")
        print(rows[-1])

    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>SnapFace samples</title></head>"
            f"<body><h1>SnapFace - échantillons ({stamp})</h1>"
            + "".join(f"<div style='display:inline-block'><p>{n}</p>{l}</div>"
                      for n, l in zip(available_effects(), links))
            + "</body></html>")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[OK] index.html écrit ({len(links)} effets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())