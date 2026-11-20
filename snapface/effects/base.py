"""Base des effets : grille d'échantillonnage normalisée + grid_sample."""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


def create_grid(h: int, w: int, device: str = "cpu") -> torch.Tensor:
    """Grille normalisée (1,2,h,w) : [0]=u (colonnes), [1]=v (lignes), [-1,1]."""
    u = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
    v = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
    return torch.cat([u, v], dim=1).contiguous()


def sample(crop: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Échantillonne crop (1,3,h,w) sur la grille (1,2,h,w)."""
    return F.grid_sample(crop, grid.permute(0, 2, 3, 1), mode="bilinear",
                         padding_mode="border", align_corners=True)


def morph_grid(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Interpole deux grilles : transition continue entre deux effets."""
    return a.lerp(b, float(min(max(t, 0.0), 1.0)))


def anchor_uv(anchors: dict | None, name: str, default: tuple = (0.5, 0.5)) -> tuple:
    """Position normalisée [0,1] d'un landmark -> coordonnées grille [-1,1]."""
    ax, ay = (anchors or {}).get(name, default)
    return 2.0 * float(ax) - 1.0, 2.0 * float(ay) - 1.0


def falloff(u, v, u0: float, v0: float, radius: float):
    """Retourne (du, dv, poids) autour d'un point d'ancrage, poids nul hors rayon."""
    du, dv = u - u0, v - v0
    r2 = (du * du + dv * dv) / (radius * radius)
    return du, dv, torch.clamp(1.0 - r2, 0.0, 1.0) ** 2


class Effect(ABC):
    """Déformation exprimée comme une fonction de la grille d'échantillonnage."""

    name: str = "base"
    defaults: dict = {}

    def __init__(self, device: str = "cpu", **params):
        self.device = device
        self.params = {**self.defaults, **params}
        self.gain = 1.0
        self._cache: dict = {}

    @abstractmethod
    def warp(self, grid: torch.Tensor, anchors: dict | None = None) -> torch.Tensor:
        """Transforme la grille normalisée (1,2,h,w) -> (1,2,h,w)."""

    def set_gain(self, gain: float) -> None:
        """Module temporellement l'amplitude (pulsation, morph)."""
        self.gain = float(gain)

    def grid_for(self, h: int, w: int, anchors: dict | None = None) -> torch.Tensor:
        """Grille déformée, mise en cache tant que la taille et les ancres sont stables."""
        key = (h, w, tuple(sorted(anchors.items())) if anchors else None)
        if self.gain == 1.0 and key in self._cache:
            return self._cache[key]
        grid = self.warp(create_grid(h, w, self.device), anchors)
        if self.gain == 1.0:
            if len(self._cache) > 6:
                self._cache.clear()
            self._cache[key] = grid
        return grid

    def apply(self, crop: torch.Tensor, anchors: dict | None = None) -> torch.Tensor:
        h, w = crop.shape[-2:]
        return sample(crop, self.grid_for(h, w, anchors))
