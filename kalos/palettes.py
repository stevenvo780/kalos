"""Curated palettes for mathematical beauty. Each maps [0,1] -> RGB.

The point is drama: deep voids, incandescent cores, saturated mid-tones.
Add your own — the grade stage accepts any matplotlib-style colormap.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# (name -> list of (stop, hex)) — hand-tuned for glow on black
_STOPS: dict[str, list[tuple[float, str]]] = {
    "aurora":   [(0.0, "#01010a"), (0.35, "#0b3d6b"), (0.6, "#1fa88f"),
                 (0.82, "#8bf3c2"), (1.0, "#f2fff4")],
    "ember":    [(0.0, "#0a0100"), (0.4, "#7a1607"), (0.68, "#e0533f"),
                 (0.88, "#ffb347"), (1.0, "#fff6d8")],
    "neon":     [(0.0, "#04010f"), (0.4, "#3a0ca3"), (0.66, "#c026d3"),
                 (0.85, "#ff4d9d"), (1.0, "#fff0fb")],
    "ice":      [(0.0, "#00030f"), (0.42, "#0d2f6b"), (0.68, "#2a7fff"),
                 (0.86, "#8fd0ff"), (1.0, "#f4fbff")],
    "gold":     [(0.0, "#050300"), (0.45, "#4d2c00"), (0.7, "#c8891a"),
                 (0.88, "#ffcf4d"), (1.0, "#fffbe6")],
    "spectral": [(0.0, "#03000a"), (0.25, "#3a0ca3"), (0.5, "#1fa88f"),
                 (0.72, "#e0c020"), (0.88, "#e0533f"), (1.0, "#fff0f0")],
    "magma":    [(0.0, "#030008"), (0.28, "#31005c"), (0.52, "#8f1d6b"),
                 (0.73, "#e34b5f"), (0.9, "#ffad67"), (1.0, "#fff7cf")],
    "plasma":   [(0.0, "#030014"), (0.3, "#3e138f"), (0.55, "#a62098"),
                 (0.75, "#ed4b69"), (0.9, "#fca636"), (1.0, "#fff6a1")],
    "deepsea":  [(0.0, "#00040a"), (0.3, "#00264d"), (0.52, "#006878"),
                 (0.72, "#00b5a6"), (0.88, "#77f4d4"), (1.0, "#effffb")],
    "toxic":    [(0.0, "#010600"), (0.32, "#073b12"), (0.56, "#168f38"),
                 (0.76, "#65d63d"), (0.91, "#d5f451"), (1.0, "#fbffd8")],
    "coral":    [(0.0, "#080103"), (0.32, "#4a1024"), (0.55, "#a52a45"),
                 (0.74, "#f05b57"), (0.9, "#ffae78"), (1.0, "#fff1d7")],
    "viridis":  [(0.0, "#02040d"), (0.28, "#35286f"), (0.5, "#167d85"),
                 (0.72, "#4fbd6b"), (0.9, "#b8dc3d"), (1.0, "#fff59a")],
    "starlight": [(0.0, "#020207"), (0.35, "#191b3f"), (0.58, "#555a9b"),
                  (0.78, "#b3b8df"), (0.92, "#e8dcb8"), (1.0, "#ffffff")],
}


@lru_cache(maxsize=None)
def cmap(name: str) -> LinearSegmentedColormap:
    if name not in _STOPS:
        raise KeyError(f"palette '{name}' not found; have {list(_STOPS)}")
    stops = _STOPS[name]
    return LinearSegmentedColormap.from_list(name, [(s, c) for s, c in stops], N=1024)


def names() -> list[str]:
    return list(_STOPS)
