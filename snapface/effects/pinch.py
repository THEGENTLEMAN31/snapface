"""Effet pinch : pincement radial, le contenu est attiré vers le centre."""
import torch
from .base import Effect, falloff


class Pinch(Effect):
    name = "pinch"
    defaults = {"strength": 0.4, "radius": 0.9}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        s = self.params["strength"] * self.gain
        du, dv, fac = falloff(u, v, 0.0, 0.0, self.params["radius"])
        k = torch.clamp(1.0 - s * fac, 0.15, 1.0)
        return torch.cat([du / k, dv / k], dim=1)
