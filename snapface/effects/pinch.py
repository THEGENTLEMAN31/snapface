"""Effet pinch : pincement radial vers le centre."""
import torch
from .base import Effect


class Pinch(Effect):
    name = "pinch"
    defaults = {"strength": 0.4}
    R = 0.9

    def warp(self, grid):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        fac = (1.0 - (r / self.R) ** 2) ** 2
        s = self.params["strength"]
        u2 = u * (1.0 + s * fac)
        v2 = v * (1.0 + s * fac)
        inside = r < self.R
        return torch.cat([torch.where(inside, u2, u), torch.where(inside, v2, v)], dim=1)