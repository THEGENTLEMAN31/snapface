"""Détection de visage par YuNet (ONNX, cv2.FaceDetectorYN)."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import YUNET_PATH


@dataclass
class Detection:
    box: np.ndarray
    score: float
    landmarks: list | None


class FaceDetector:
    """YuNet en avant-plan : boîte la plus confiante + 5 landmarks.

    Format de sortie OpenCV 5 : [x, y, w, h, 5*(lx, ly), score]
    (le score est la DERNIÈRE colonne, contrairement aux anciennes versions).
    """

    DETECT_WIDTH: int = 480
    SCORE_COL: int = 14

    def __init__(self, conf_threshold: float = 0.6):
        self._using = "none"
        self._scale = 1.0
        self._detector = None
        self.conf_threshold = float(conf_threshold)
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_PATH), config="", input_size=(320, 320),
                score_threshold=self.conf_threshold, nms_threshold=0.3, top_k=5000)
            self._using = "yunet"
        except Exception:
            self._detector = None

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        """Retourne la détection la plus confiante, ou None."""
        if self._detector is None:
            return None
        h, w = frame_bgr.shape[:2]
        self._scale = 1.0
        src = frame_bgr
        if w > self.DETECT_WIDTH:
            self._scale = self.DETECT_WIDTH / w
            src = cv2.resize(frame_bgr, (self.DETECT_WIDTH, int(round(h * self._scale))))
        sh, sw = src.shape[:2]
        self._detector.setInputSize((sw, sh))
        _, dets = self._detector.detect(src)
        if dets is None or len(dets) == 0:
            return None
        best = dets[np.argmax(dets[:, self.SCORE_COL])]
        x, y, bw, bh = (float(v) for v in best[:4])
        score = float(best[self.SCORE_COL])
        pts = best[4:4 + 10].reshape(5, 2).astype(float)
        if self._scale != 1.0:
            inv = 1.0 / self._scale
            x, y, bw, bh = x * inv, y * inv, bw * inv, bh * inv
            pts *= inv
        landmarks = [(float(px), float(py)) for px, py in pts]
        return Detection(box=np.array([x, y, bw, bh], dtype=float),
                         score=score, landmarks=landmarks)

    @property
    def using(self) -> str:
        return self._using
