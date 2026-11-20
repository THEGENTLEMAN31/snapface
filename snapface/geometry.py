"""Conversion cv2<->torch et géométrie des crops/bboxes."""
import cv2
import numpy as np
import torch


def to_tensor(bgr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """bgr (H,W,3) uint8 -> tensor (1,3,H,W) float32 [0,1] sur device."""
    t = torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1))).to(device)
    return t.unsqueeze(0).to(torch.float32) / 255.0


def from_tensor(t: torch.Tensor) -> np.ndarray:
    """tensor (1,3,H,W) -> numpy (H,W,3) uint8, chan dernier, clip 0..255."""
    arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def clamp_bbox(box, frame_shape) -> tuple:
    """Retourne (x, y, w, h) borné dans frame_shape (h, w)."""
    x, y, w, h = box
    H, W = frame_shape[:2]
    x, y = float(max(x, 0)), float(max(y, 0))
    w = float(min(w, W - x))
    h = float(min(h, H - y))
    return x, y, w, h


def square_box(box, frame_shape, expand: float = 0.35) -> tuple:
    """Box élargie, de côté carré, recentrée puis rabattue dans le frame."""
    x, y, w, h = (float(v) for v in box)
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * expand)
    H, W = frame_shape[:2]
    side = min(side, float(min(H, W)))
    bx = min(max(cx - side / 2.0, 0.0), W - side)
    by = min(max(cy - side / 2.0, 0.0), H - side)
    return bx, by, side, side


def crop_face(frame: np.ndarray, det, expand: float = 0.35):
    """Découpe une région carrée autour du visage : (crop, box dans le frame)."""
    box = det.box if hasattr(det, "box") else det
    bx, by, side, _ = square_box(box, frame.shape, expand)
    x0, y0 = int(round(bx)), int(round(by))
    x1, y1 = int(round(bx + side)), int(round(by + side))
    crop = frame[y0:y1, x0:x1]
    return crop, (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


_MASK_CACHE: dict[tuple, np.ndarray] = {}


def feather_mask(h: int, w: int, feather: int = 15, shape: str = "ellipse") -> np.ndarray:
    """Masque (h,w) float32 : 1 au centre, 0 sur le bord, transition smoothstep."""
    key = (h, w, feather, shape)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    b = max(1, min(int(feather), max(min(h, w) // 2, 1)))
    if shape == "rect":
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        dy = np.minimum(ys, h - 1 - ys) / b
        dx = np.minimum(xs, w - 1 - xs) / b
        d = np.minimum(dx, dy)
    else:
        ny = (np.arange(h, dtype=np.float32) / max(h - 1, 1)) * 2.0 - 1.0
        nx = (np.arange(w, dtype=np.float32) / max(w - 1, 1)) * 2.0 - 1.0
        r = np.sqrt(nx[None, :] ** 2 + ny[:, None] ** 2)
        d = (1.0 - r) * (0.5 * min(h, w) / b)
    t = np.clip(d, 0.0, 1.0)
    mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    if len(_MASK_CACHE) > 16:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = mask
    return mask


def soft_mask(h: int, w: int, feather_frac: float = 0.35, device: str = "cpu") -> torch.Tensor:
    """Même masque rectangulaire que feather_mask, en tensor (1,1,h,w) sur device."""
    b = max(1, int(round(min(h, w) * feather_frac)))
    return torch.from_numpy(feather_mask(h, w, b, shape="rect")).to(device).view(1, 1, h, w)


def feather_mask_cached(h: int, w: int, feather: int = 15,
                        shape: str = "ellipse") -> np.ndarray:
    """Masque de fusion, mémorisé par taille quantifiée (la bbox varie chaque frame)."""
    qh, qw = max(16, (h // 16) * 16), max(16, (w // 16) * 16)
    base = feather_mask(qh, qw, feather, shape)
    if (qh, qw) == (h, w):
        return base
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)


def paste_face(frame: np.ndarray, face_img: np.ndarray, box, feather: int = 15,
               shape: str = "ellipse") -> np.ndarray:
    """Recolle face_img (BGR) dans frame à la position box, alpha prémultiplié."""
    x, y, w, h = (int(round(v)) for v in box)
    if w <= 0 or h <= 0:
        return frame
    H, W = frame.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    rw, rh = min(w, W - x), min(h, H - y)
    if rw <= 1 or rh <= 1:
        return frame
    if face_img.shape[1] != rw or face_img.shape[0] != rh:
        face_img = cv2.resize(face_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    m = feather_mask_cached(rh, rw, feather, shape)[:, :, None]
    roi = frame[y:y + rh, x:x + rw]
    out = face_img.astype(np.float32)
    out -= roi
    out *= m
    out += roi
    np.clip(out, 0, 255, out=out)
    frame[y:y + rh, x:x + rw] = out.astype(np.uint8)
    return frame


def landmarks_to_norm(landmarks, box) -> np.ndarray | None:
    """Landmarks en pixels -> coordonnées normalisées [0,1] dans la bbox."""
    if landmarks is None or len(landmarks) == 0:
        return None
    x, y, w, h = box
    pts = np.asarray(landmarks, dtype=np.float32)
    return (pts - np.array([x, y], np.float32)) / np.array([max(w, 1.0), max(h, 1.0)],
                                                           np.float32)


def anchors_in_crop(anchors: np.ndarray | None, box, crop_box) -> dict | None:
    """Landmarks normalisés dans la bbox -> coordonnées normalisées dans le crop."""
    if anchors is None:
        return None
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop_box
    pts = np.asarray(anchors, dtype=np.float32) * np.array([bw, bh], np.float32) \
        + np.array([bx, by], np.float32)
    pts = (pts - np.array([cx, cy], np.float32)) / np.array([max(cw, 1.0), max(ch, 1.0)],
                                                            np.float32)
    from .tracking import LM_NAMES
    return {name: (float(u), float(v)) for name, (u, v) in zip(LM_NAMES, pts)}


def seam_error(before: np.ndarray, after: np.ndarray, box, ring: int = 4) -> float:
    """Écart max sur l'anneau de bord de la zone recollée (mesure de couture)."""
    x, y, w, h = (int(round(v)) for v in box)
    H, W = before.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    if w < 2 * ring + 2 or h < 2 * ring + 2:
        return 0.0
    m = np.zeros((h, w), bool)
    m[:ring, :] = m[-ring:, :] = m[:, :ring] = m[:, -ring:] = True
    a = before[y:y + h, x:x + w][m].astype(np.int16)
    b = after[y:y + h, x:x + w][m].astype(np.int16)
    return float(np.abs(a - b).max())
