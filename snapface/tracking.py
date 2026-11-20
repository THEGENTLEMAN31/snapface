"""Suivi temporel du visage : One Euro + suivi Lucas-Kanade des landmarks."""
import cv2
import numpy as np
from .detector import Detection
from .geometry import landmarks_to_norm

# indices YuNet : oeil droit, oeil gauche, nez, coin bouche droit, coin bouche gauche
LM_NAMES = ("eye_r", "eye_l", "nose", "mouth_r", "mouth_l")


class OneEuro:
    """Filtre One Euro scalaire (Casiez et al.) : peu de latence quand ça bouge vite."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.06, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t, self._dx = float(x), float(t), 0.0
            return float(x)
        dt = max(t - self._t, 1e-4)
        raw_dx = (x - self._x) / dt
        adx = self._dx + self._alpha(self.d_cutoff, dt) * (raw_dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(adx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        self._dx, self._t = adx, t
        return self._x


def _box_of(pts: np.ndarray, prev_box: tuple) -> tuple:
    """Estime (x,y,w,h) par similarité (translation + échelle inter-oculaire)."""
    x, y, w, h = prev_box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cx, cy, w, h


class BoxTracker:
    """Lisse la bbox et la propage par flux optique entre deux détections."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.06,
                 expire: float = 0.6, hold_scale: bool = True):
        self.filters = {k: OneEuro(min_cutoff, beta) for k in ("cx", "cy", "w", "h")}
        self.expire = expire
        self.hold_scale = hold_scale
        self.box: tuple | None = None
        self.anchors: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._last_seen: float = -1e9
        self._last_fresh: float = -1e9
        self.stats = {"detections": 0, "tracked": 0, "lost": 0}

    def reset(self) -> None:
        self.box, self.anchors = None, None
        self._prev_pts = self._prev_gray = None
        for f in self.filters.values():
            f._x = None

    def _emit(self, cx: float, cy: float, w: float, h: float, t: float) -> tuple:
        f = self.filters
        cx, cy = f["cx"](cx, t), f["cy"](cy, t)
        w, h = f["w"](w, t), f["h"](h, t)
        w = h = max((w + h) / 2.0, 16.0)
        self.box = (cx - w / 2.0, cy - h / 2.0, w, h)
        return self.box

    def update(self, det: Detection | None, gray: np.ndarray, t: float) -> tuple | None:
        fresh = det is not None
        if fresh:
            x, y, w, h = (float(v) for v in det.box)
            self._emit(x + w / 2.0, y + h / 2.0, w, h, t)
            self.anchors = landmarks_to_norm(det.landmarks, (x, y, w, h))
            self._prev_pts = np.array(det.landmarks, dtype=np.float32) if det.landmarks else None
            self._prev_gray = gray
            self._last_seen = self._last_fresh = t
            self.stats["detections"] += 1
            return self.box

        if self.box is None or t - self._last_seen > self.expire:
            if self.box is not None:
                self.stats["lost"] += 1
            self.reset()
            return None

        moved = False
        if self._prev_pts is not None and self._prev_gray is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None,
                                                  winSize=(21, 21), maxLevel=3)
            nxt = nxt.reshape(-1, 2)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, nxt, None,
                                                   winSize=(21, 21), maxLevel=3)
            back = back.reshape(-1, 2)
            err = np.linalg.norm(back - self._prev_pts, axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (err < 1.5)
            if ok.sum() >= 3:
                p0, p1 = self._prev_pts[ok], nxt[ok]
                shift = (p1 - p0).mean(axis=0)
                x, y, w, h = self.box
                self.anchors = (self._prev_pts[ok] - np.array([x, y])) / max(w, 1.0)
                self._prev_pts = nxt
                scale = 1.0
                if self.hold_scale and ok[0] and ok[1]:
                    d0, d1 = np.linalg.norm(p0[0] - p0[1]), np.linalg.norm(p1[0] - p1[1])
                    if d0 > 1.0:
                        scale = float(np.clip(d1 / d0, 0.97, 1.03))
                cx, cy = x + w / 2.0 + shift[0], y + h / 2.0 + shift[1]
                self._emit(cx, cy, w * scale, h * scale, t)
                moved = True
                self.stats["tracked"] += 1

        self._prev_gray = gray
        return self.box


def anchors_dict(anchors: np.ndarray | None) -> dict | None:
    """Convertit les 5 landmarks normalisés en dict nommé pour les effets."""
    if anchors is None:
        return None
    return {name: (float(a), float(b)) for name, (a, b) in zip(LM_NAMES, anchors)}
