"""Effet kaleidoscope : repli angulaire en secteurs symétriques."""
import torch
from .base import Effect


class Kaleidoscope(Effect):
    name = "kaleidoscope"
    defaults = {"segments": 6, "spin": 0.0}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        theta = torch.atan2(v, u) + self.params["spin"]
        sectors = max(int(self.params["segments"]), 2)
        sector = torch.pi / sectors
        folded = torch.abs(torch.remainder(theta, 2.0 * sector) - sector)
        return torch.cat([r * torch.cos(folded), r * torch.sin(folded)], dim=1)
