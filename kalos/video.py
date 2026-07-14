"""Encode a sequence of RGB frames into an Instagram-ready reel (9:16 mp4)."""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

REEL_W, REEL_H = 1080, 1920  # 9:16 vertical


class Reel:
    def __init__(self, out_path: str | Path, fps: int = 30, tmp: str | Path | None = None):
        self.out = Path(out_path)
        self.fps = fps
        self.tmp = Path(tmp or (self.out.parent / f".frames_{self.out.stem}"))
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.i = 0

    def add(self, rgb: np.ndarray) -> None:
        img = Image.fromarray(rgb, "RGB")
        if img.size != (REEL_W, REEL_H):
            img = img.resize((REEL_W, REEL_H), Image.LANCZOS)
        img.save(self.tmp / f"f{self.i:05d}.png")
        self.i += 1

    def encode(self, crf: int = 18) -> Path:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(self.fps),
            "-i", str(self.tmp / "f%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        for f in self.tmp.glob("f*.png"):
            f.unlink()
        self.tmp.rmdir()
        return self.out
