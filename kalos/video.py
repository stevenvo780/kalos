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
        # frames are saved at their native (possibly half) res; encode() scales
        # to 1080x1920 with lanczos so grain gets averaged on the way up.
        Image.fromarray(rgb, "RGB").save(self.tmp / f"f{self.i:05d}.png")
        self.i += 1

    def encode(self, crf: int = 26, denoise: str = "hqdn3d=2:1.5:6:6") -> Path:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        vf = ["scale=%d:%d:flags=lanczos" % (REEL_W, REEL_H)]
        if denoise:
            vf.append(denoise)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(self.fps),
            "-i", str(self.tmp / "f%05d.png"),
            "-vf", ",".join(vf),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        for f in self.tmp.glob("f*.png"):
            f.unlink()
        self.tmp.rmdir()
        return self.out
