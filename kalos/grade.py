"""Color grading: turn a scalar density/height field into a graded RGB frame.

This is the "filtros de color" stage: palette map + bloom/glow + gamma +
vignette. Tuned to make math read as *spectacular*, not as a plot.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .palettes import cmap


def normalize(field: np.ndarray, log: bool = True, p_lo: float = 0.0,
              p_hi: float = 99.8) -> np.ndarray:
    f = field.astype("float32")
    if log:
        f = np.log1p(f)
    lo, hi = np.percentile(f, [p_lo, p_hi])
    return np.clip((f - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _vignette(shape, strength=0.35):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    return np.clip(1.0 - strength * (r ** 2), 0.0, 1.0)


def grade(field: np.ndarray, palette: str = "aurora", *, log: bool = True,
          gamma: float = 0.85, bloom: float = 1.4, bloom_sigma: float = 7.0,
          vignette: float = 0.35, p_hi: float = 99.8) -> np.ndarray:
    """field -> uint8 RGB (H,W,3). bloom adds a blurred glow of bright zones."""
    n = normalize(field, log=log, p_hi=p_hi)
    if gamma != 1.0:
        n = np.power(n, gamma)
    rgb = cmap(palette)(n)[..., :3].astype("float32")  # (H,W,3) in [0,1]
    if bloom > 0:
        glow = gaussian_filter(n, sigma=bloom_sigma)
        glow = (glow / (glow.max() + 1e-9)) ** 1.3
        rgb = rgb + bloom * glow[..., None] * cmap(palette)(0.92)[:3]
    if vignette > 0:
        rgb *= _vignette(rgb.shape[:2], vignette)[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype("uint8")
