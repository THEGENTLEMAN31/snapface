"""Écriture vidéo : pas de temps conforme au temps réel, ré-encodage H.264 final."""
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class FrameWriter:
    """Écrit chaque frame rendue puis rectifie la cadence à la fermeture."""

    CODECS = ("avc1", "mp4v")

    def __init__(self, path: Path, fps: float, size: tuple[int, int],
                 export_h264: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        self.export_h264 = export_h264
        self.raw = self.path.with_name(self.path.stem + ".raw.mp4")
        self._w = None
        self._codec = None
        self._n = 0
        for cc in self.CODECS:
            w = cv2.VideoWriter(str(self.raw), cv2.VideoWriter_fourcc(*cc),
                                self.fps, self.size)
            if w.isOpened():
                self._w, self._codec = w, cc
                break
            w.release()
        if self._w is None:
            raise ValueError(f"Aucun codec vidéo disponible pour {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        self._w.write(frame)
        self._n += 1

    def release(self, actual_fps: float | None = None) -> Path:
        """Ferme, rectifie la cadence réelle et ré-encode en H.264 si ffmpeg est là."""
        self._w.release()
        n = self._n
        rate = actual_fps if actual_fps and actual_fps > 1.0 else self.fps
        ffmpeg = shutil.which("ffmpeg")
        if self.export_h264 and ffmpeg and n > 1:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-r", f"{rate:.3f}",
                   "-i", str(self.raw),
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(self.path)]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                self.raw.unlink(missing_ok=True)
                return self.path
        os.replace(self.raw, self.path)
        return self.path

    @property
    def codec(self) -> str:
        return self._codec or "none"
