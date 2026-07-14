"""Harmonograph: damped coupled pendulums trace luminous Lissajous weaves.

A slow phase drift makes the figure breathe; the trail is rendered as a
soft-additive density so grade.py can bloom it into neon filament art.
"""
from __future__ import annotations
import numpy as np


def frames(n_frames=300, size=(1920, 1080), samples=90_000,
           freqs=(2.0, 3.0, 3.0, 2.0), damp=0.0018, seed=3):
    H, W = size
    aspect = W / H
    f1, f2, f3, f4 = freqs
    tt = np.linspace(0, 60 * np.pi, samples).astype("float32")
    env = np.exp(-damp * tt)
    for i in range(n_frames):
        t = i / n_frames
        ph = 2 * np.pi * t
        x = env * (np.sin(f1 * tt + ph) + np.sin(f2 * tt + ph * 1.3))
        y = env * (np.sin(f3 * tt + ph * 0.7) + np.sin(f4 * tt + ph))
        r = 3.2
        dens, _, _ = np.histogram2d(
            y, x, bins=[H, W], range=[(-r, r), (-r * aspect, r * aspect)])
        yield dens.astype("float32")
