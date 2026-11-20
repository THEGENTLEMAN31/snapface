"""Effet mirror : le côté droit est réfléchi sur le côté gauche."""
import torch
from .base import Effect


class Mirror(Effect):
    name = "mirror"

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        return torch.cat([-torch.abs(u), v], dim=1)
