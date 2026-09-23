"""Effet identity : aucun changement (témoin)."""
from .base import Effect


class Identity(Effect):
    name = "identity"

    def warp(self, grid):
        return grid