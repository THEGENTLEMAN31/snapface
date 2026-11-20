"""Effet identity : témoin, la grille n'est pas modifiée."""
from .base import Effect


class Identity(Effect):
    name = "identity"

    def warp(self, grid, anchors=None):
        return grid
