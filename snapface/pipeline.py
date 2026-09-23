"""Pipeline : capture -> détection -> effet PyTorch -> affichage/enregistrement."""
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Config, YUNET_PATH
from .detector import FaceDetector
from .effects import get_effect, available_effects
from .geometry import to_tensor, from_tensor, crop_face, paste_face, clamp_bbox

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Pipeline:
    def __init__(self, cfg: Config, source: str = "webcam",
                 image_path: str | None = None, video_path: str | None = None):
        self.cfg = cfg
        self.source = source
        self.image_path = image_path
        self.video_path = video_path
        self.device = self._pick_device()
        if cfg.effect not in available_effects():
            raise ValueError(f"Effet inconnu: {cfg.effect} (disponibles: {available_effects()})")
        self.effects = {n: get_effect(n, device=self.device, params=cfg.effect_params)
                        for n in available_effects()}
        self.effect = self.effects[cfg.effect]
        self.effect_idx = available_effects().index(cfg.effect)
        self.detector = FaceDetector(conf_threshold=cfg.conf_threshold)
        self.smooth_bbox: tuple | None = None
        self.last_det_time: float = -1e9
        self.fps = 0.0

    def _pick_device(self) -> str:
        if self.cfg.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.cfg.device

    # --- boucle principale -------------------------------------------------
    def run(self) -> int:
        if self.source == "image":
            return self._run_image()
        return self._run_stream()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        alpha = self.cfg.smooth_alpha
        now = time.monotonic()
        det = self.detector.detect(frame)
        if det is not None:
            new_box = tuple(float(v) for v in det.box)
            x, y, w, h = new_box
            if self.smooth_bbox is not None and alpha < 1.0 and w > 0 and h > 0:
                ox, oy, ow, oh = self.smooth_bbox
                if ow > 0 and oh > 0:
                    new_box = (ox + (x - ox) * alpha,
                               oy + (y - oy) * alpha,
                               ow + (w - ow) * alpha,
                               oh + (h - oh) * alpha)
            self.smooth_bbox = clamp_bbox(new_box, frame.shape)
            self.last_det_time = now
        elif self.smooth_bbox is not None and now - self.last_det_time > 0.5:
            self.smooth_bbox = None

        if self.smooth_bbox is None or self.cfg.preview:
            return frame

        crop, box = crop_face(frame, self.smooth_bbox)
        if crop.size == 0:
            return frame
        crop_r = cv2.resize(crop, (self.cfg.crop_size, self.cfg.crop_size))
        t = to_tensor(crop_r, self.device)
        warped = self.effect.apply(t)
        face = from_tensor(warped)
        face = cv2.resize(face, (box[2], box[3]))
        return paste_face(frame, face, box, feather=15)

    def _put_overlay(self, frame: np.ndarray) -> None:
        if self.cfg.clean:
            return
        label = f"SnapFace - {self.effect.name} - {self.fps:.0f} FPS"
        cv2.putText(frame, label, (12, 32), _FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # --- modes -------------------------------------------------------------
    def _run_image(self) -> int:
        img = cv2.imread(self.image_path) if self.image_path else None
        if img is None:
            print(f"Impossible de lire l'image: {self.image_path}")
            return 1
        frame = img.copy()
        alpha_bak = self.cfg.smooth_alpha
        self.cfg.smooth_alpha = 1.0
        frame = self._process_frame(frame)
        self.cfg.smooth_alpha = alpha_bak
        self._put_overlay(frame)
        if self.cfg.record is not None:
            Path(self.cfg.record).parent.mkdir(parents=True, exist_ok=True)
            ext = Path(self.cfg.record).suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                cv2.imwrite(str(self.cfg.record), frame)
                print(f"[OK] image écrite: {self.cfg.record}")
            else:
                writer = self._make_writer_image(frame)
                if writer is None:
                    return 1
                for _ in range(max(int(self.cfg.fps_target * 2), 1)):
                    writer.write(frame)
                writer.release()
                print(f"[OK] vidéo écrite: {self.cfg.record}")
        else:
            src = Path(self.image_path) if self.image_path else Path("in")
            out = Path("out") / f"{src.stem}_{self.effect.name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"[OK] image écrite: {out}")
        if not self.cfg.clean:
            if self._show(frame, wait=True) is None:
                print("Aucun affichage possible (pas de DISPLAY ?) — image traitée et écrite quand même.")
        return 0

    def _show(self, frame: np.ndarray, wait: bool = False) -> int | None:
        """Affiche frame, retourne la touche lue, ou None sans display."""
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None
        try:
            cv2.namedWindow("SnapFace", cv2.WINDOW_NORMAL)
            cv2.imshow("SnapFace", frame)
            return (cv2.waitKey(0) or 0) & 0xFF if wait else (cv2.waitKey(1) & 0xFF)
        except cv2.error:
            cv2.destroyAllWindows()
            return None

    def _run_stream(self) -> int:
        cap, is_file = self._open_source()
        if cap is None:
            if is_file:
                print(f"Impossible d'ouvrir la vidéo: {self.video_path}")
            else:
                print(f"Caméra introuvable (indice {self.cfg.cam_index}). "
                      f"Vérifiez /dev/video*, ou utilisez --source image --image photo.jpg")
            return 1
        writer = None
        try:
            writer = self._make_writer(cap) if self.cfg.record else None
            demo_idx = self.effect_idx
            t_start = time.monotonic()
            key = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._resize_input(frame)
                now = time.monotonic()
                if self.cfg.demo_seconds and now - t_start >= self.cfg.demo_seconds:
                    t_start = now
                    demo_idx = (demo_idx + 1) % len(self.effects)
                    self.effect = list(self.effects.values())[demo_idx]
                frame = self._process_frame(frame)
                self.fps = self._measure_fps()
                if writer is not None:
                    writer.write(frame)
                if not self.cfg.clean:
                    display = frame.copy()
                    self._put_overlay(display)
                    ret = self._show(display, wait=False)
                    if ret is None:
                        print("Aucun affichage possible (pas de DISPLAY ?) — mode sans fenêtre.")
                        self.cfg.clean = True
                        key = 0
                    else:
                        key = ret
                if key in (ord('q'), 27):
                    break
                if ord('1') <= key <= ord('6'):
                    idx = min(key - ord('1'), len(self.effects) - 1)
                    self.effect = list(self.effects.values())[idx]
                    demo_idx = idx
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        return 0

    def _measure_fps(self) -> float:
        now = time.monotonic()
        dt = now - getattr(self, "_t", now)
        self._t = now
        return 1.0 / max(1e-6, dt)

    # --- sources -----------------------------------------------------------
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

    def _resize_input(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != (self.cfg.input_w, self.cfg.input_h):
            return cv2.resize(frame, (self.cfg.input_w, self.cfg.input_h))
        return frame

    def _make_writer_image(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        Path(self.cfg.record).parent.mkdir(parents=True, exist_ok=True)
        fps = self.cfg.fps_target
        for fourcc_str in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(self.cfg.record), cv2.VideoWriter_fourcc(*fourcc_str),
                                     float(fps), (int(w), int(h)))
            if writer.isOpened():
                return writer
        print(f"Échec: aucun codec vidéo disponible pour {self.cfg.record}.")
        return None

    def _make_writer(self, cap):
        w, h = self.cfg.input_w, self.cfg.input_h
        fps = self.cfg.fps_target if self.source == "webcam" else cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 30
        Path(self.cfg.record).parent.mkdir(parents=True, exist_ok=True)
        for fourcc_str in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(self.cfg.record), cv2.VideoWriter_fourcc(*fourcc_str),
                                     float(fps), (int(w), int(h)))
            if writer.isOpened():
                print(f"[OK] enregistrement MP4 ({fourcc_str}) -> {self.cfg.record}")
                return writer
        print(f"Échec: aucun codec vidéo disponible pour {self.cfg.record}.")
        return None