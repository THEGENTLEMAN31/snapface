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
    using: str = "none"

    def __init__(self, conf_threshold: float = 0.6):
        self._detector = None
        self.using = "none"
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_PATH),
                config="",
                input_size=(320, 320),
                score_threshold=conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.using = "yunet"
        except Exception:
            self._detector = None

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        if self._detector is None:
            return None
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        rows, dets = self._detector.detect(frame_bgr)
        if dets is None or len(dets) == 0:
            return None
        best = dets[dets[:, 4].argmax()]
        x, y, bw, bh, score = best[:5].astype(float)
        landmarks = []
        for i in range(5):
            landmarks.append((best[5 + 2 * i], best[6 + 2 * i]))
        return Detection(box=np.array([x, y, bw, bh], dtype=float),
                         score=float(score),
                         landmarks=landmarks)

    @property
    def using(self) -> str:
        return self._using

    @using.setter
    def using(self, value: str):
        self._using = value