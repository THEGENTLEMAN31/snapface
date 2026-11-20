"""Banc d'essai : temps par étage, GPU vs CPU, débit réel."""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from snapface.config import Config
from snapface.detector import FaceDetector
from snapface.effects import available_effects, get_effect
from snapface.geometry import from_tensor, paste_face, to_tensor
from snapface.pipeline import Pipeline
from snapface.tracking import BoxTracker

TMP = Path("/tmp/opencode/sftest")


def _photo(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path)
        if img is not None:
            return img
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / "lena.jpg"
    if not dst.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        urllib.request.urlretrieve(url, dst)
    return cv2.imread(str(dst))


def _stats(xs: list[float]) -> tuple[float, float]:
    return statistics.median(xs), float(np.percentile(xs, 90))


def bench_stages(device: str, image_path: str | None, crop_size: int,
                 iters: int) -> dict:
    img = _photo(image_path)
    frame = cv2.resize(img, (1280, 720))
    detector = FaceDetector(conf_threshold=0.6)
    det = detector.detect(frame)
    box = (480.0, 260.0, 320.0, 320.0) if det is None else tuple(float(v) for v in det.box)
    crop = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
    crop = cv2.resize(crop, (crop_size, crop_size))
    effect = get_effect("swirl", device=device)
    acc = {k: [] for k in ("detect", "to_tensor", "warp", "from_tensor", "paste")}

    detector.detect(frame)
    for _ in range(iters):
        t = time.perf_counter()
        detector.detect(frame)
        acc["detect"].append(time.perf_counter() - t)

        t = time.perf_counter()
        tensor = to_tensor(crop, device)
        acc["to_tensor"].append(time.perf_counter() - t)

        t = time.perf_counter()
        grid = effect.grid_for(crop_size, crop_size, None)
        from snapface.effects.base import sample
        warped = sample(tensor, grid)
        if device == "cuda":
            torch.cuda.synchronize()
        acc["warp"].append(time.perf_counter() - t)

        t = time.perf_counter()
        face = from_tensor(warped)
        acc["from_tensor"].append(time.perf_counter() - t)

        t = time.perf_counter()
        paste_face(frame.copy(), face, box, feather=15)
        acc["paste"].append(time.perf_counter() - t)

    res = {k: dict(zip(("median_ms", "p90_ms"),
                       (1000 * _stats(v)[0], 1000 * _stats(v)[1]))) for k, v in acc.items()}
    total = sum(r["median_ms"] for r in res.values())
    res["total"] = {"median_ms": total, "p90_ms": sum(r["p90_ms"] for r in res.values())}
    res["fps_max"] = 1000.0 / total if total else 0.0
    return res


def bench_stream(device: str, video: Path, frames: int, crop_size: int) -> dict:
    cfg = Config(device=device, clean=True, crop_size=crop_size, det_every=2,
                 effect="swirl", input_w=1280, input_h=720)
    pipe = Pipeline(cfg, source="video", video_path=str(video))
    cap = cv2.VideoCapture(str(video))
    times, dets = [], 0
    t_start = time.perf_counter()
    i = 0
    while i < frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (1280, 720))
        t = time.perf_counter()
        _, info = pipe.process(frame, i / 30.0)
        times.append(time.perf_counter() - t)
        dets += int(info["box"] is not None)
        i += 1
    cap.release()
    elapsed = time.perf_counter() - t_start
    return {"frames": frames, "fps": frames / elapsed,
            "median_ms": 1000 * statistics.median(times),
            "p90_ms": 1000 * float(np.percentile(times, 90)),
            "frames_avec_visage_pct": 100.0 * dets / max(frames, 1)}


def bench_tracker(video: Path, frames: int, smooth: bool) -> dict:
    """Jitter de la bbox, avec ou sans lissage One Euro (mesure A/B)."""
    det = FaceDetector(0.6)
    tr = BoxTracker(min_cutoff=1.2 if smooth else 1e6, beta=0.06 if smooth else 0.0)
    cap = cv2.VideoCapture(str(video))
    cs, ws, lost = [], [], 0
    i = 0
    while i < frames:
        ok, frame = cap.read()
        if not ok:
            break
        d = det.detect(frame)
        box = tr.update(d, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), i / 30.0)
        i += 1
        if box is None:
            lost += 1
            continue
        cs.append((box[0] + box[2] / 2.0, box[1] + box[3] / 2.0))
        ws.append(box[2])
    cap.release()
    c = np.array(cs, float)
    if len(c) < 3:
        return {"frames": i, "jitter_px": 0.0, "taille_cv_pct": 0.0, "perdues": lost}
    accel = np.abs(np.diff(c, n=2, axis=0))
    return {
        "frames": i,
        "jitter_px": float(np.mean(np.linalg.norm(accel, axis=1))),
        "taille_cv_pct": float(100.0 * np.std(ws) / max(np.mean(ws), 1e-6)),
        "perdues": lost,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench SnapFace : ms par étage, GPU vs CPU.")
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=str(TMP / "trailer.mp4"))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    devices = [d for d in ("cuda", "cpu")
               if d == "cpu" or torch.cuda.is_available()]
    report = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "crop_size": args.crop_size, "devices": {}}
    for dev in devices:
        print(f"=== {dev.upper()} ===")
        stages = bench_stages(dev, args.image, args.crop_size, args.iters)
        print(f"{'etage':14s} {'median':>8s} {'p90':>8s}")
        for name, s in stages.items():
            if name in ("total", "fps_max"):
                continue
            print(f"{name:14s} {s['median_ms']:7.2f}ms {s['p90_ms']:7.2f}ms")
        print(f"{'TOTAL':14s} {stages['total']['median_ms']:7.2f}ms "
              f"-> {stages['fps_max']:.1f} fps max")
        entry = {"stages": stages}
        if Path(args.video).exists():
            stream = bench_stream(dev, Path(args.video), args.frames, args.crop_size)
            entry["stream"] = stream
            print(f"flux video 1280x720 : {stream['fps']:.1f} fps, "
                  f"{stream['median_ms']:.2f} ms/frame, "
                  f"visage sur {stream['frames_avec_visage_pct']:.0f}% des frames")
            raw = bench_tracker(Path(args.video), args.frames, smooth=False)
            smooth = bench_tracker(Path(args.video), args.frames, smooth=True)
            entry["stabilite"] = {"sans_lissage": raw, "avec_lissage": smooth}
            print(f"jitter bbox : {raw['jitter_px']:.2f} px sans lissage -> "
                  f"{smooth['jitter_px']:.2f} px avec (x{raw['jitter_px'] / max(smooth['jitter_px'], 1e-6):.1f})")
            print(f"taille bbox : {raw['taille_cv_pct']:.2f}% -> {smooth['taille_cv_pct']:.2f}% (CV)")
        report["devices"][dev] = entry
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] rapport JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
