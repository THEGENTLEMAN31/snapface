"""Effet bulge : gonflement radial autour du centre (ou de chaque ancre)."""
import torch
from .base import Effect, falloff


class Bulge(Effect):
    name = "bulge"
    defaults = {"strength": 0.35, "radius": 0.9}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        s = self.params["strength"] * self.gain
        du, dv, fac = falloff(u, v, 0.0, 0.0, self.params["radius"])
        return torch.cat([u - s * fac * du, v - s * fac * dv], dim=1)
