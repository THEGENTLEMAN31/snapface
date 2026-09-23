"""Effet swirl : tourbillon angulaire décroissant du centre."""
import torch
from .base import Effect


class Swirl(Effect):
    name = "swirl"
    defaults = {"swirl_strength": 1.4}
    R = 1.0

    def warp(self, grid):
        u, v = grid[:, 0:1], grid[:, 1:2]
        r = torch.sqrt(u * u + v * v)
        theta = torch.atan2(v, u)
        s = self.params["swirl_strength"]
        theta2 = theta + s * (1.0 - r / self.R)
        u2 = r * torch.cos(theta2)
        v2 = r * torch.sin(theta2)
        return torch.cat([u2, v2], dim=1)