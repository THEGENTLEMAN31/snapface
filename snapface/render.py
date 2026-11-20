"""Habitacle : HUD, maillage de déformation, split avant/après."""
import cv2
import numpy as np
import torch

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
YELLOW = (0, 230, 255)
DARK = (24, 24, 24)


def _text(img, s: str, xy: tuple, scale: float = 0.6, color=GREEN, thickness: int = 2) -> None:
    cv2.putText(img, s, xy, FONT, scale, DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, effect: str, fps: float, device: str,
             extra: str = "") -> None:
    """Bandeau haut-gauche : effet, fps, device."""
    _text(frame, f"SnapFace - {effect} - {fps:.0f} FPS - {device.upper()}", (14, 34), 0.8)
    if extra:
        _text(frame, extra, (14, 64), 0.55, YELLOW)


def draw_grid(frame: np.ndarray, grid: torch.Tensor, box, step: int = 8) -> None:
    """Dessine le maillage d'échantillonnage déformé par-dessus la zone traitée."""
    x, y, w, h = box
    g = grid[0].detach().cpu().numpy()
    gh, gw = g.shape[1], g.shape[2]
    step = max(2, min(step, gh // 2))
    px = x + (g[0] + 1.0) * 0.5 * w
    py = y + (g[1] + 1.0) * 0.5 * h
    for i in range(0, gh, step):
        pts = np.stack([px[i], py[i]], axis=1).round().astype(np.int32)
        cv2.polylines(frame, [pts], False, YELLOW, 1, cv2.LINE_AA)
    for j in range(0, gw, step):
        pts = np.stack([px[:, j], py[:, j]], axis=1).round().astype(np.int32)
        cv2.polylines(frame, [pts], False, YELLOW, 1, cv2.LINE_AA)


def draw_split(frame: np.ndarray, raw: np.ndarray) -> None:
    """Gauche : image brute. Droite : image traitée, séparées par un trait."""
    h, w = frame.shape[:2]
    mid = w // 2
    frame[:, :mid] = raw[:, :mid]
    cv2.line(frame, (mid, 0), (mid, h), (255, 255, 255), 2, cv2.LINE_AA)
    _text(frame, "brut", (mid - 90, h - 18), 0.6)
    _text(frame, "traite", (mid + 16, h - 18), 0.6)


def draw_box(frame: np.ndarray, box, color=(0, 160, 255)) -> None:
    """Rectangle de la zone traitée."""
    x, y, w, h = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
