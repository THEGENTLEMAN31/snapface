"""Conversion cv2<->torch et math des crops/bboxes."""
import cv2
import numpy as np
import torch


def to_tensor(bgr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """bgr (H,W,3) uint8 -> tensor (1,3,H,W) float32 [0,1] sur device."""
    t = torch.from_numpy(bgr.transpose(2, 0, 1)).to(device)
    return t.unsqueeze(0).to(torch.float32) / 255.0


def from_tensor(t: torch.Tensor) -> np.ndarray:
    """tensor (1,3,H,W) -> numpy (H,W,3) uint8, chan dernier, clip 0..255."""
    arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def soft_mask(h: int, w: int, feather_frac: float = 0.35, device: str = "cpu") -> torch.Tensor:
    """Masque (1,1,H,W) float32 [0,1] : 1 au centre, 0 sur les bords."""
    y = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
    x = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
    dx = torch.abs(x) / 1.0
    dy = torch.abs(y) / 1.0
    d = torch.maximum(dx, dy)
    t = torch.clamp(d * (1.0 / feather_frac), 0.0, 1.0)
    return (1.0 + torch.cos(t * np.pi)) / 2.0


def clamp_bbox(box, frame_shape):
    """Retourne (x, y, w, h) borné dans frame_shape (h, w)."""
    x, y, w, h = box
    H, W = frame_shape[:2]
    x, y = float(max(x, 0)), float(max(y, 0))
    w = float(min(w, W - x))
    h = float(min(h, H - y))
    return x, y, w, h


def crop_face(frame: np.ndarray, det, expand: float = 0.35):
    """Découpe la région du visage avec marges, retourne (crop, box ajustée)."""
    box = det.box if hasattr(det, "box") else det
    x, y, w, h = clamp_bbox(box, frame.shape)
    ex, ey = w * expand, h * expand
    cx0, cy0 = max(0, x - ex), max(0, y - ey)
    cx1, cy1 = min(frame.shape[1], x + w + ex), min(frame.shape[0], y + h + ey)
    crop = frame[int(round(cy0)):int(round(cy1)), int(round(cx0)):int(round(cx1))]
    return crop, (round(cx0), round(cy0), round(cx1 - cx0), round(cy1 - cy0))


def paste_face(frame, face_img, box, feather: int = 15) -> np.ndarray:
    """Recolle face_img (BGR) dans frame à la position box avec fusion gaussienne."""
    x, y, w, h = (int(round(v)) for v in box)
    if w <= 0 or h <= 0:
        return frame
    face = cv2.resize(face_img, (w, h))
    H, W = frame.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    rw = min(w, W - x)
    rh = min(h, H - y)
    if rw <= 0 or rh <= 0:
        return frame
    face = face[:rh, :rw]
    feather = max(1, min(feather, min(rw, rh) // 2))
    mask = np.zeros((rh, rw), dtype=np.float32)
    mask[:] = 1.0
    if feather > 0:
        cv2.GaussianBlur(mask, (0, 0), feather / 2.0, dst=mask)
    roi = frame[y:y + rh, x:x + rw]
    m = mask[:, :, None]
    frame[y:y + rh, x:x + rw] = (face.astype(np.float32) * m
                                  + roi.astype(np.float32) * (1.0 - m))
    return frame