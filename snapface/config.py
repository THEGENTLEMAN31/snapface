"""Configuration globale et chemins des assets."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
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
    # v2
    det_every: int = 2
    min_cutoff: float = 1.2
    beta: float = 0.06
    debug_grid: bool = False
    split: bool = False
    pulse: float = 0.0
    morph_seconds: float = 0.45
    seconds: float | None = None
    export_h264: bool = True
    feather: int = 15
    review: bool = False
