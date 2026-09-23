"""Base des effets de déformation géométrique."""
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


def create_grid(H: int, W: int, device: str = "cpu") -> torch.Tensor:
    """Grille normalisée (1,2,H,W) : [0]=u (colonnes), [1]=v (lignes), valeurs [-1,1]."""
    u = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
    v = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
    return torch.cat([u, v], dim=1)


class Effect(ABC):
    """Effet de déformation : grille (1,2,H,W) modifiée puis grid_sample."""

    name: str = "base"
    defaults: dict = {}

    def __init__(self, device: str = "cpu", **params):
        self.device = device
        self.params = {**self.defaults, **params}
        self._dim_key: tuple | None = None
        self._base_grid: torch.Tensor | None = None

    @abstractmethod
    def warp(self, grid: torch.Tensor) -> torch.Tensor:
        """Modifie la grille d'échantillonnage normalisée (1,2,H,W) -> (1,2,H,W)."""

    def _grid(self, H: int, W: int) -> torch.Tensor:
        if self._dim_key != (H, W):
            self._base_grid = create_grid(H, W, self.device)
            self._dim_key = (H, W)
        return self._base_grid

    def apply(self, crop: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) crop tensor -> (1,3,H,W) warped."""
        H, W = crop.shape[-2:]
        warped = self.warp(self._grid(H, W))
        g4 = warped.permute(0, 2, 3, 1)
        return F.grid_sample(crop, g4, mode="bilinear", padding_mode="border", align_corners=True)