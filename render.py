#!/usr/bin/env python3
"""Render a kalos scene to a 9:16 Instagram reel.

Usage:
    python render.py <scene> [seconds] [fps] [out.mp4]
    python render.py --list

Scenes are a registry of (frame generator, color-grade recipe). Adding a new
piece = add a generator under kalos/scenes/ and one entry to SCENES.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from kalos.video import Reel, REEL_W, REEL_H
from kalos.grade import grade
from kalos.scenes import attractor, harmonograph

# internal render grid (H, W); upscaled to 1080x1920 on encode
GRID = (REEL_H, REEL_W)

SCENES = {
    "attractor-dejong": dict(
        gen=lambda nf: attractor.frames("de_jong", nf, GRID, points=1_500_000),
        grade=dict(palette="aurora", bloom=1.5, bloom_sigma=8, gamma=0.8)),
    "attractor-clifford": dict(
        gen=lambda nf: attractor.frames("clifford", nf, GRID, points=1_500_000,
                                        base=(-1.7, 1.8, -1.9, -0.4),
                                        amp=(0.12, 0.12, 0.1, 0.1)),
        grade=dict(palette="neon", bloom=1.6, bloom_sigma=9, gamma=0.82)),
    "harmonograph": dict(
        gen=lambda nf: harmonograph.frames(nf, GRID),
        grade=dict(palette="ice", bloom=1.7, bloom_sigma=6, gamma=0.9)),
}


def render_scene(name: str, seconds: float = 12, fps: int = 30,
                 out: str | None = None) -> Path:
    if name not in SCENES:
        raise SystemExit(f"unknown scene '{name}'. have: {list(SCENES)}")
    spec = SCENES[name]
    n_frames = int(round(seconds * fps))
    out = Path(out or f"out/{name}.mp4")
    reel = Reel(out, fps=fps)
    t0 = time.time()
    for i, field in enumerate(spec["gen"](n_frames)):
        reel.add(grade(field, **spec["grade"]))
        if i % 30 == 0:
            print(f"  {name}: frame {i}/{n_frames}  ({time.time()-t0:.0f}s)", flush=True)
    path = reel.encode()
    print(f"OK {name} -> {path}  ({time.time()-t0:.0f}s, {n_frames} frames)")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l"):
        print("scenes:", ", ".join(SCENES)); sys.exit(0)
    name = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    out = sys.argv[4] if len(sys.argv) > 4 else None
    render_scene(name, secs, fps, out)
