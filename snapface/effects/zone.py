"""Effets ancrés sur les landmarks : gros yeux et grande bouche."""
import torch
from .base import Effect, anchor_uv, falloff


class BigEyes(Effect):
    name = "big_eyes"
    defaults = {"strength": 0.55, "radius": 0.30}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        s = self.params["strength"] * self.gain
        acc_u = torch.zeros_like(u)
        acc_v = torch.zeros_like(v)
        for name in ("eye_l", "eye_r"):
            u0, v0 = anchor_uv(anchors, name)
            du, dv, fac = falloff(u, v, u0, v0, self.params["radius"])
            acc_u = acc_u - s * fac * du
            acc_v = acc_v - s * fac * dv
        return torch.cat([u + acc_u, v + acc_v], dim=1)


class BigMouth(Effect):
    name = "big_mouth"
    defaults = {"strength": 0.5, "radius": 0.34}

    def warp(self, grid, anchors=None):
        u, v = grid[:, 0:1], grid[:, 1:2]
        s = self.params["strength"] * self.gain
        u0, v0 = anchor_uv(anchors, "mouth_l")
        u1, _ = anchor_uv(anchors, "mouth_r")
        du, dv, fac = falloff(u, v, (u0 + u1) / 2.0, v0, self.params["radius"])
        return torch.cat([u - s * fac * du, v - 0.6 * s * fac * dv], dim=1)
