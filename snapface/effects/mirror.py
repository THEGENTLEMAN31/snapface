"""Effet mirror : symétrie horizontale, moitié droite reflétée sur la gauche."""
import torch
from .base import Effect


class Mirror(Effect):
    name = "mirror"

    def warp(self, grid):
        u = grid[:, 0:1]
        u2 = torch.where(u > 0, -u, u)
        return torch.cat([u2, grid[:, 1:2]], dim=1)