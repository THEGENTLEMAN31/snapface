"""Registre des effets : enregistrement par décorateur et fabrique."""
from .base import Effect
from .identity import Identity
from .bulge import Bulge
from .pinch import Pinch
from .swirl import Swirl
from .mirror import Mirror
from .kaleidoscope import Kaleidoscope
from .zone import BigEyes, BigMouth

_EFFECTS: dict[str, type[Effect]] = {}

for _cls in (Identity, Bulge, Pinch, Swirl, Mirror, Kaleidoscope, BigEyes, BigMouth):
    _EFFECTS[_cls.name] = _cls


def register(cls: type[Effect]) -> type[Effect]:
    """Décorateur : enregistre une classe d'effet sous son nom."""
    _EFFECTS[cls.name] = cls
    return cls


def available_effects() -> list[str]:
    """Noms d'effets dans l'ordre des touches 1..8."""
    return list(_EFFECTS)


def get_effect(name: str, device: str = "cpu", params: dict | None = None) -> Effect:
    """Instancie un effet par nom."""
    if name not in _EFFECTS:
        raise ValueError(f"Effet inconnu: {name} (disponibles: {available_effects()})")
    return _EFFECTS[name](device=device, **(params or {}))
