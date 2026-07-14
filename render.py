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
from kalos.scenes import (
    attractor,
    automata,
    biological,
    fractals,
    harmonograph,
    physical,
    reaction,
    strange,
)

# internal render grid (H, W); upscaled to 1080x1920 on encode.
# half-res averages histogram grain (smoother + far smaller files); LANCZOS
# upscale keeps the glow crisp.
GRID = (REEL_H // 2, REEL_W // 2)


def _model(module, kind: str, **recipe):
    """Build a registry spec while keeping each catalog entry declarative."""
    return dict(
        gen=lambda nf: module.frames(kind, nf, GRID),
        grade=recipe,
    )

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

    # Strange attractors: continuous flows and iterated maps.
    "attractor-lorenz": _model(
        strange, "lorenz", palette="aurora", bloom=1.55, bloom_sigma=7,
        gamma=0.80, p_hi=99.8),
    "attractor-aizawa": _model(
        strange, "aizawa", palette="neon", bloom=1.50, bloom_sigma=8,
        gamma=0.82, p_hi=99.7),
    "attractor-rossler": _model(
        strange, "rossler", palette="ember", bloom=1.65, bloom_sigma=7,
        gamma=0.78, p_hi=99.7),
    "attractor-thomas": _model(
        strange, "thomas", palette="ice", bloom=1.45, bloom_sigma=6,
        gamma=0.84, p_hi=99.7),
    "attractor-halvorsen": _model(
        strange, "halvorsen", palette="spectral", bloom=1.50, bloom_sigma=8,
        gamma=0.82, p_hi=99.7),
    "attractor-dadras": _model(
        strange, "dadras", palette="gold", bloom=1.60, bloom_sigma=7,
        gamma=0.78, p_hi=99.7),
    "attractor-chen": _model(
        strange, "chen", palette="neon", bloom=1.55, bloom_sigma=8,
        gamma=0.82, p_hi=99.8),
    "attractor-gumowski-mira": _model(
        strange, "gumowski_mira", palette="aurora", bloom=1.50,
        bloom_sigma=7, gamma=0.80, p_hi=99.7),
    "attractor-hopalong": _model(
        strange, "hopalong", palette="gold", bloom=1.55, bloom_sigma=7,
        gamma=0.76, p_hi=99.7),
    "attractor-ikeda": _model(
        strange, "ikeda", palette="ice", bloom=1.70, bloom_sigma=7,
        gamma=0.76, p_hi=99.5),
    "attractor-tinkerbell": _model(
        strange, "tinkerbell", palette="ember", bloom=1.65, bloom_sigma=6,
        gamma=0.78, p_hi=99.5),
    "attractor-svensson": _model(
        strange, "svensson", palette="spectral", bloom=1.45, bloom_sigma=7,
        gamma=0.82, p_hi=99.8),
    "attractor-bedhead": _model(
        strange, "bedhead", palette="neon", bloom=1.55, bloom_sigma=7,
        gamma=0.80, p_hi=99.7),
    "attractor-duffing": _model(
        strange, "duffing", palette="ember", bloom=1.55, bloom_sigma=6,
        gamma=0.82, p_hi=99.8),
    "attractor-standard-map": _model(
        strange, "standard_map", palette="ice", bloom=1.35, bloom_sigma=6,
        gamma=0.86, p_hi=99.8),

    # Escape-time and stability fractals.
    "fractal-mandelbrot": _model(
        fractals, "mandelbrot", palette="spectral", bloom=1.15,
        bloom_sigma=8, gamma=0.78, p_hi=99.7),
    "fractal-julia": _model(
        fractals, "julia", palette="neon", bloom=1.15, bloom_sigma=8,
        gamma=0.82, p_hi=99.7),
    "fractal-burning-ship": _model(
        fractals, "burning_ship", palette="ember", bloom=1.05,
        bloom_sigma=8, gamma=0.82, p_hi=99.6),
    "fractal-newton": _model(
        fractals, "newton", palette="ice", bloom=1.25, bloom_sigma=6,
        gamma=0.78, p_hi=99.5),
    "fractal-lyapunov": _model(
        fractals, "lyapunov", palette="aurora", bloom=1.05, bloom_sigma=6,
        gamma=0.78, p_hi=99.4),
    "fractal-phoenix": _model(
        fractals, "phoenix", palette="gold", bloom=1.00, bloom_sigma=8,
        gamma=0.83, p_hi=99.7),
    "fractal-multibrot": _model(
        fractals, "multibrot", palette="plasma", bloom=1.10, bloom_sigma=8,
        gamma=0.80, p_hi=99.7),

    # Discrete and continuous life-like automata.
    "life-conway": _model(
        automata, "conway", palette="neon", bloom=1.35, bloom_sigma=5,
        gamma=0.82, p_hi=99.6),
    "life-lenia": _model(
        automata, "lenia", palette="spectral", log=False, bloom=1.10,
        bloom_sigma=5, gamma=0.92, p_hi=99.6),
    "life-smoothlife": _model(
        automata, "smoothlife", palette="aurora", log=False, bloom=1.15,
        bloom_sigma=5, gamma=0.90, p_hi=99.7),
    "ca-brians-brain": _model(
        automata, "brians_brain", palette="ice", bloom=1.45,
        bloom_sigma=5, gamma=0.82, p_hi=99.4),
    "ca-cyclic": _model(
        automata, "cyclic_ca", palette="spectral", log=False, bloom=0.80,
        bloom_sigma=4, gamma=0.95, p_hi=99.8),
    "ca-rule110": _model(
        automata, "rule110", palette="gold", log=False, bloom=0.40,
        bloom_sigma=3, gamma=1.05, p_hi=99.8),
    "ca-rule30": _model(
        automata, "rule30", palette="neon", log=False, bloom=0.55,
        bloom_sigma=3, gamma=1.00, p_hi=99.8),
    "ca-langtons-ant": _model(
        automata, "langtons_ant", palette="ember", bloom=1.20,
        bloom_sigma=5, gamma=0.85, p_hi=99.5),

    # Reaction-diffusion and pattern-forming PDEs.
    "reaction-gray-scott": _model(
        reaction, "gray_scott", palette="aurora", log=False, bloom=1.25,
        bloom_sigma=5, gamma=0.88, p_hi=99.6),
    "reaction-fitzhugh-nagumo": _model(
        reaction, "fitzhugh_nagumo", palette="ember", log=False, bloom=1.25,
        bloom_sigma=5, gamma=0.88, p_hi=99.5),
    "reaction-turing": _model(
        reaction, "turing", palette="gold", log=False, bloom=0.75,
        bloom_sigma=4, gamma=0.95, p_hi=99.6),
    "reaction-belousov-zhabotinsky": _model(
        reaction, "belousov_zhabotinsky", palette="spectral", log=False,
        bloom=1.00, bloom_sigma=5, gamma=0.90, p_hi=99.7),
    "reaction-swift-hohenberg": _model(
        reaction, "swift_hohenberg", palette="ice", log=False, bloom=0.90,
        bloom_sigma=4, gamma=0.95, p_hi=99.7),

    # Biological aggregation and growth.
    "growth-physarum": _model(
        biological, "physarum", palette="toxic", log=False, bloom=0.50,
        bloom_sigma=7, gamma=0.90, p_hi=99.8),
    "growth-dla": _model(
        biological, "dla", palette="ice", bloom=0.35, bloom_sigma=7,
        gamma=0.95, p_hi=99.95),
    "growth-eden": _model(
        biological, "eden", palette="coral", log=False, bloom=0.80,
        bloom_sigma=7, gamma=0.85, p_hi=99.7),
    "growth-barnsley-fern": _model(
        biological, "barnsley_fern", palette="toxic", bloom=0.80,
        bloom_sigma=6, gamma=0.80, p_hi=99.7),
    "growth-lsystem-plant": _model(
        biological, "lsystem_plant", palette="deepsea", bloom=0.90,
        bloom_sigma=6, gamma=0.85, p_hi=99.7),

    # Spreading, mechanics, waves, curves and statistical physics.
    "spread-sir": _model(
        physical, "sir_epidemic", palette="plasma", log=False, bloom=0.45,
        bloom_sigma=7, gamma=0.95, p_hi=99.8),
    "spread-forest-fire": _model(
        physical, "forest_fire", palette="ember", log=False, bloom=0.70,
        bloom_sigma=6, gamma=0.90, p_hi=99.8),
    "spread-percolation": _model(
        physical, "percolation", palette="viridis", log=False, bloom=0.55,
        bloom_sigma=5, gamma=0.90, p_hi=99.8),
    "physics-wave-interference": _model(
        physical, "wave_interference", palette="deepsea", log=False,
        bloom=0.25, bloom_sigma=5, gamma=1.00, p_hi=99.8),
    "physics-chladni": _model(
        physical, "chladni", palette="gold", log=False, bloom=0.65,
        bloom_sigma=5, gamma=0.82, p_hi=99.8),
    "physics-double-pendulum": _model(
        physical, "double_pendulum", palette="neon", bloom=0.80,
        bloom_sigma=6, gamma=0.85, p_hi=99.7),
    "physics-nbody": _model(
        physical, "nbody", palette="deepsea", bloom=0.30,
        bloom_sigma=7, gamma=1.00, p_hi=99.85),
    "physics-curl-flow": _model(
        physical, "curl_flow", palette="deepsea", log=False, bloom=0.25,
        bloom_sigma=5, gamma=1.15, p_hi=99.9),
    "pattern-phyllotaxis": _model(
        physical, "phyllotaxis", palette="gold", log=False, bloom=0.15,
        bloom_sigma=5, gamma=1.70, p_hi=99.9),
    "curve-maurer-rose": _model(
        physical, "maurer_rose", palette="neon", bloom=0.70,
        bloom_sigma=5, gamma=0.90, p_hi=99.7),
    "curve-fourier-epicycles": _model(
        physical, "fourier_epicycles", palette="coral", bloom=1.00,
        bloom_sigma=5, gamma=0.80, p_hi=99.7),
    "physics-ising": _model(
        physical, "ising", palette="magma", log=False, bloom=0.35,
        bloom_sigma=5, gamma=1.00, p_hi=99.85),
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
