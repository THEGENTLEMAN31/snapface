"""Pipeline : capture -> suivi -> effet PyTorch -> fusion -> affichage/enregistrement."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import render
from .config import Config
from .detector import FaceDetector
from .effects import available_effects, get_effect
from .effects.base import morph_grid, sample
from .geometry import anchors_in_crop, crop_face, from_tensor, paste_face, to_tensor
from .tracking import BoxTracker
from .writer import FrameWriter

KEYS = [str(i) for i in range(1, 10)]


class Pipeline:
    def __init__(self, cfg: Config, source: str = "webcam",
                 image_path: str | None = None, video_path: str | None = None):
        self.cfg = cfg
        self.source = source
        self.image_path = image_path
        self.video_path = video_path
        self.device = self._pick_device()
        names = available_effects()
        if cfg.effect not in names:
            raise ValueError(f"Effet inconnu: {cfg.effect} (disponibles: {names})")
        self.effects = {n: get_effect(n, device=self.device, params=cfg.effect_params)
                        for n in names}
        self.order = names
        self.demo_order = [n for n in names if n != "identity"]
        self.effect = self.effects[cfg.effect]
        self._prev_effect = None
        self._morph_t0 = -1e9
        smooth = cfg.smooth_alpha < 1.0
        self.tracker = BoxTracker(min_cutoff=cfg.min_cutoff if smooth else 1e6,
                                  beta=cfg.beta if smooth else 0.0)
        self.detector = FaceDetector(conf_threshold=cfg.conf_threshold)
        self.timings = {"detect": deque(maxlen=60), "warp": deque(maxlen=60),
                        "paste": deque(maxlen=60), "loop": deque(maxlen=60)}
        self.fps = 0.0
        self._t0 = time.monotonic()
        self._frame_idx = 0
        self.raw_frame: np.ndarray | None = None

    def _pick_device(self) -> str:
        if self.cfg.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.cfg.device

    # --- traitement d'une frame -------------------------------------------
    def process(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, dict]:
        """Retourne (frame traité, infos {grid, box, anchors})."""
        self.raw_frame = frame
        info: dict = {"grid": None, "box": None, "anchors": None}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        det = None
        if self.cfg.det_every <= 1 or self._frame_idx % max(1, self.cfg.det_every) == 0:
            det = self.detector.detect(frame)
        self.timings["detect"].append(time.perf_counter() - t0)

        box = self.tracker.update(det, gray, time.monotonic())
        if box is None or self.cfg.preview:
            return frame, info
        crop, cbox = crop_face(frame, box, expand=0.35)
        if crop.size == 0:
            return frame, info
        anchors = anchors_in_crop(self.tracker.anchors, box, cbox)

        if self.cfg.pulse > 0.0:
            gain = 1.0 + self.cfg.pulse * math.sin(2.0 * math.pi * t / 2.0)
            self.effect.set_gain(gain)
            if self._prev_effect is not None:
                self._prev_effect.set_gain(gain)

        t1 = time.perf_counter()
        tensor = to_tensor(cv2.resize(crop, (self.cfg.crop_size, self.cfg.crop_size)),
                           self.device)
        grid = self._grid(tensor, anchors, t)
        warped = sample(tensor, grid)
        face = from_tensor(warped)
        self.timings["warp"].append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        out = paste_face(frame, face, cbox, feather=self.cfg.feather)
        self.timings["paste"].append(time.perf_counter() - t2)
        info.update(grid=grid, box=cbox, anchors=anchors)
        return out, info

    def _grid(self, tensor, anchors, t: float):
        """Grille de l'effet courant, interpolée avec la précédente pendant un morph."""
        h, w = tensor.shape[-2:]
        cur = self.effect.grid_for(h, w, anchors)
        if self._prev_effect is None:
            return cur
        k = 1.0 if self.cfg.morph_seconds <= 0 else (t - self._morph_t0) / self.cfg.morph_seconds
        if k >= 1.0:
            self._prev_effect = None
            return cur
        prev = self._prev_effect.grid_for(h, w, anchors)
        return morph_grid(prev, cur, k)

    # --- boucle ------------------------------------------------------------
    def run(self) -> int:
        if self.source == "image":
            return self._run_image()
        return self._run_stream()

    def _run_image(self) -> int:
        img = cv2.imread(self.image_path) if self.image_path else None
        if img is None:
            print(f"Impossible de lire l'image: {self.image_path}")
            return 1
        out, info = self.process(img.copy(), 0.0)
        if info["grid"] is not None and self.cfg.debug_grid:
            render.draw_grid(out, info["grid"], info["box"])
        target = self.cfg.record
        if target is None:
            src = Path(self.image_path)
            target = Path("out") / f"{src.stem}_{self.effect.name}.png"
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            cv2.imwrite(str(target), out)
        else:
            wr = FrameWriter(target, self.cfg.fps_target, (out.shape[1], out.shape[0]))
            for _ in range(max(int(self.cfg.fps_target * 2), 1)):
                wr.write(out)
            wr.release()
        print(f"[OK] image écrite: {target}")
        if not self.cfg.clean:
            shown = out.copy()
            render.draw_hud(shown, self.effect.name, 0.0, self.device)
            self._show(shown, wait=True)
        return 0

    def _run_stream(self) -> int:
        cap, is_file = self._open_source()
        if cap is None:
            if is_file:
                print(f"Impossible d'ouvrir la vidéo: {self.video_path}")
            else:
                print(f"Caméra introuvable (indice {self.cfg.cam_index}). "
                      f"Vérifiez /dev/video*, ou utilisez --source image --image photo.jpg")
            return 1
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        writer = None
        n_written = 0
        demo_names = self.demo_order
        demo_idx = demo_names.index(self.effect.name) if self.effect.name in demo_names else 0
        next_demo = self.cfg.demo_seconds or 0.0
        try:
            self._t0 = time.monotonic()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self._resize_input(frame)
                wall = time.monotonic() - self._t0
                t = wall if self.source == "webcam" else self._frame_idx / max(src_fps, 1.0)
                if self.cfg.seconds is not None and \
                        (wall if self.source == "webcam" else t) >= self.cfg.seconds:
                    break
                if self.cfg.demo_seconds and t >= next_demo:
                    next_demo += self.cfg.demo_seconds
                    demo_idx = (demo_idx + 1) % len(demo_names)
                    self.switch(demo_names[demo_idx], t)
                out, info = self.process(frame, t)
                if info["grid"] is not None and self.cfg.debug_grid:
                    render.draw_grid(out, info["grid"], info["box"])
                    render.draw_box(out, info["box"])
                if self.cfg.split and self.raw_frame is not None:
                    render.draw_split(out, frame)
                if writer is None and self.cfg.record is not None:
                    writer = FrameWriter(self.cfg.record, self._writer_fps(src_fps),
                                         (out.shape[1], out.shape[0]), self.cfg.export_h264)
                    print(f"[OK] enregistrement ({writer.codec}) -> {self.cfg.record}")
                if writer is not None:
                    writer.write(out)
                    n_written += 1
                self._frame_idx += 1
                self.fps = self._tick_fps()
                if self.cfg.clean:
                    if self._check_stop_key(cv2.waitKey(1) & 0xFF if self._has_display() else 0xFF):
                        break
                    continue
                display = out.copy()
                render.draw_hud(display, self.effect.name, self.fps, self.device,
                                self._hud_extra())
                key = self._show(display)
                if key is None:
                    self.cfg.clean = True
                    print("Aucun affichage possible (pas de DISPLAY ?) — mode sans fenêtre.")
                    continue
                if self._check_stop_key(key):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if writer is not None:
                path = writer.release(self.fps if self.source == "webcam" else None)
                rate = self.fps if self.source == "webcam" else max(src_fps, 1.0)
                print(f"[OK] vidéo finalisée: {path} ({n_written} frames, "
                      f"{n_written / max(rate, 1e-6):.1f} s)")
        return 0

    def switch(self, name: str, t: float) -> None:
        """Change d'effet avec transition morph depuis l'effet courant."""
        new = self.effects[name]
        if new is self.effect:
            return
        self._prev_effect = self.effect
        self._morph_t0 = t
        self.effect = new

    def _check_stop_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True
        ch = chr(key) if 0 < key < 128 else ""
        if ch in KEYS:
            idx = KEYS.index(ch)
            if idx < len(self.order):
                self.switch(self.order[idx], self._now())
        elif ch == "d":
            self.cfg.debug_grid = not self.cfg.debug_grid
        elif ch == "s":
            self.cfg.split = not self.cfg.split
        return False

    # --- utilitaires -------------------------------------------------------
    def _now(self) -> float:
        return (time.monotonic() - self._t0) if self.source == "webcam" \
            else self._frame_idx / max(self.cfg.fps_target, 1)

    def _tick_fps(self) -> float:
        self.timings["loop"].append(time.monotonic())
        ts = list(self.timings["loop"])
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)

    def _hud_extra(self) -> str:
        if not self.timings["warp"]:
            return ""
        d = 1000 * float(np.median(self.timings["detect"]))
        w = 1000 * float(np.median(self.timings["warp"]))
        p = 1000 * float(np.median(self.timings["paste"]))
        return f"detect {d:.1f} ms | warp {w:.1f} ms | paste {p:.1f} ms"

    def _writer_fps(self, src_fps: float) -> float:
        if self.source == "video" and 1.0 < src_fps <= 120:
            return src_fps
        return float(self.cfg.fps_target)

    def _has_display(self) -> bool:
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def _show(self, frame: np.ndarray, wait: bool = False) -> int | None:
        """Affiche frame, retourne la touche lue, ou None sans display."""
        if not self._has_display():
            return None
        try:
            cv2.namedWindow("SnapFace", cv2.WINDOW_NORMAL)
            cv2.imshow("SnapFace", frame)
            return (cv2.waitKey(0) or 0) & 0xFF if wait else (cv2.waitKey(1) & 0xFF)
        except cv2.error:
            cv2.destroyAllWindows()
            return None

    def _resize_input(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != (self.cfg.input_w, self.cfg.input_h):
            return cv2.resize(frame, (self.cfg.input_w, self.cfg.input_h))
        return frame

    def _open_source(self):
        if self.source == "video":
            cap = cv2.VideoCapture(self.video_path)
            return (cap, True) if cap.isOpened() else (None, True)
        cap = cv2.VideoCapture(self.cfg.cam_index)
        if not cap.isOpened():
            return None, False
        for w, h in ((self.cfg.input_w, self.cfg.input_h), (960, 540), (640, 480)):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
                break
        return cap, False
