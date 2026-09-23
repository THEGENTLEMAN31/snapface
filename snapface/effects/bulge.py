"""Effet bulge : gonflement radial autour du centre."""
import torch
from .base import Effect


class Bulge(Effect):
    name = "bulge"
    defaults = {"strength": 0.35}
    R = 0.9

    def warp(self, grid):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        fac = (1.0 - (r / self.R) ** 2) ** 2
        s = self.params["strength"]
        u2 = u * (1.0 - s * fac)
        v2 = v * (1.0 - s * fac)
        inside = r < self.R
        return torch.cat([torch.where(inside, u2, u), torch.where(inside, v2, v)], dim=1)