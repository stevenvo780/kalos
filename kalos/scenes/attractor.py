"""Strange attractors as glowing density fields, morphing over a seamless loop.

The De Jong / Clifford maps trace millions of points into filament structures.
We accumulate them into a density grid per frame and let grade.py set them alight.
"""
from __future__ import annotations
import numpy as np


def de_jong(x, y, a, b, c, d):
    return np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)


def clifford(x, y, a, b, c, d):
    return np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)


_MAPS = {"de_jong": de_jong, "clifford": clifford}


def frames(kind="de_jong", n_frames=300, size=(1920, 1080), points=1_500_000,
           base=(1.4, -2.3, 2.4, -2.1), amp=(0.22, 0.18, 0.18, 0.22),
           phase=(0.0, 1.7, 3.1, 4.6), seed=7):
    """Yield (n_frames) density arrays of shape `size` (H, W).

    Params sweep as base+amp*sin(2*pi*t+phase) so t=0 and t=1 coincide -> loop.
    """
    step = _MAPS[kind]
    H, W = size
    aspect = W / H
    yr = (-3.6, 3.6)
    xr = (-3.6 * aspect, 3.6 * aspect)
    M = 20_000
    K = max(1, points // M)
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        t = i / n_frames
        a = base[0] + amp[0] * np.sin(2 * np.pi * t + phase[0])
        b = base[1] + amp[1] * np.sin(2 * np.pi * t + phase[1])
        c = base[2] + amp[2] * np.sin(2 * np.pi * t + phase[2])
        d = base[3] + amp[3] * np.sin(2 * np.pi * t + phase[3])
        x = rng.uniform(-1, 1, M).astype("float32")
        y = rng.uniform(-1, 1, M).astype("float32")
        xs = np.empty(K * M, dtype="float32")
        ys = np.empty(K * M, dtype="float32")
        for k in range(K):
            x, y = step(x, y, a, b, c, d)
            xs[k * M:(k + 1) * M] = x
            ys[k * M:(k + 1) * M] = y
        dens, _, _ = np.histogram2d(ys, xs, bins=[H, W], range=[yr, xr])
        yield dens.astype("float32")
