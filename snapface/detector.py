"""Détection de visage par YuNet (ONNX, cv2.FaceDetectorYN)."""
import cv2
import numpy as np
from dataclasses import dataclass
from .config import YUNET_PATH


@dataclass
class Detection:
    box: np.ndarray
    score: float
    landmarks: list | None


class FaceDetector:
    DETECT_WIDTH: int = 480

    def __init__(self, conf_threshold: float = 0.6):
        self._using = "none"
        self._scale = 1.0
        self._detector = None
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_PATH),
                config="",
                input_size=(320, 320),
                score_threshold=conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            self._using = "yunet"
        except Exception:
            self._detector = None

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        if self._detector is None:
            return None
        h, w = frame_bgr.shape[:2]
        self._scale = 1.0
        src = frame_bgr
        if w > self.DETECT_WIDTH:
            self._scale = self.DETECT_WIDTH / w
            src = cv2.resize(frame_bgr, (self.DETECT_WIDTH, int(round(h * self._scale))))
        h, w = src.shape[:2]
        self._detector.setInputSize((w, h))
        rows, dets = self._detector.detect(src)
        if dets is None or len(dets) == 0:
            return None
        best = dets[dets[:, 4].argmax()]
        x, y, bw, bh, score = best[:5].astype(float)
        if self._scale != 1.0:
            x, y, bw, bh = x / self._scale, y / self._scale, bw / self._scale, bh / self._scale
        landmarks = []
        for i in range(5):
            lx, ly = best[5 + 2 * i], best[6 + 2 * i]
            if self._scale != 1.0:
                lx, ly = lx / self._scale, ly / self._scale
            landmarks.append((float(lx), float(ly)))
        return Detection(box=np.array([x, y, bw, bh], dtype=float),
                         score=float(score),
                         landmarks=landmarks)

    @property
    def using(self) -> str:
        return self._using