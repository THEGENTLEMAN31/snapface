"""Effet kaleidoscope : repli angulaire du visage en secteurs symétriques."""
import torch
from .base import Effect


class Kaleidoscope(Effect):
    name = "kaleidoscope"
    defaults = {"segments": 6}

    def warp(self, grid):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        theta = torch.atan2(v, u)
        sectors = max(int(self.params["segments"]), 2)
        sector = torch.pi / sectors
        theta_folded = torch.abs(torch.remainder(theta, 2.0 * sector) - sector)
        u2 = r * torch.cos(theta_folded)
        v2 = r * torch.sin(theta_folded)
        return torch.cat([u2, v2], dim=1)