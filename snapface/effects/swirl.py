"""Effet swirl : rotation de plus en plus forte vers le centre."""
import torch
from .base import Effect


class Swirl(Effect):
    name = "swirl"
    defaults = {"swirl_strength": 1.4, "radius": 1.0}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        theta = torch.atan2(v, u)
        amount = self.params["swirl_strength"] * self.gain
        theta2 = theta + amount * torch.clamp(1.0 - r / self.params["radius"], 0.0, 1.0)
        return torch.cat([r * torch.cos(theta2), r * torch.sin(theta2)], dim=1)
