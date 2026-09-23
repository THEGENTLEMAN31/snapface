"""Registre des effets de déformation."""
from .base import Effect
from .identity import Identity
from .bulge import Bulge
from .pinch import Pinch
from .swirl import Swirl
from .mirror import Mirror
from .kaleidoscope import Kaleidoscope

_EFFECTS: dict[str, type[Effect]] = {}
_ORDER = ["identity", "bulge", "pinch", "swirl", "mirror", "kaleidoscope"]


def register(cls):
    _EFFECTS[cls.name] = cls
    return cls


for _cls in (Identity, Bulge, Pinch, Swirl, Mirror, Kaleidoscope):
    register(_cls)


def get_effect(name: str, device: str = "cpu", params: dict | None = None) -> Effect:
    if name not in _EFFECTS:
        raise ValueError(f"Effet inconnu: {name} (disponibles: {available_effects()})")
    return _EFFECTS[name](device=device, **(params or {}))


def available_effects() -> list[str]:
    return [n for n in _ORDER if n in _EFFECTS]