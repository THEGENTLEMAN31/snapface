"""Constantes et configuration SnapFace."""
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
YUNET_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"


@dataclass
class Config:
    cam_index: int = 0
    input_w: int = 1280
    input_h: int = 720
    crop_size: int = 256
    effect: str = "identity"
    device: str = "auto"
    demo_seconds: float | None = None
    clean: bool = False
    record: Path | None = None
    smooth_alpha: float = 0.45
    conf_threshold: float = 0.6
    effect_params: dict = field(default_factory=dict)
    preview: bool = False
    fps_target: int = 30