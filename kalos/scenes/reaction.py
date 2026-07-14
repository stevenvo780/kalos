"""Reaction--diffusion and pattern-forming PDE scenes.

All solvers use periodic boundaries on a low-resolution working lattice.  The
returned fields are continuous (with a short phosphor-like memory), finite and
nonnegative; color, bloom and vignette remain the responsibility of grade.py.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


_KINDS = {
    "gray_scott", "fitzhugh_nagumo", "turing",
    "belousov_zhabotinsky", "swift_hohenberg",
}


def _logical_shape(size: tuple[int, int], max_h: int = 240) -> tuple[int, int]:
    H, W = map(int, size)
    if H < 1 or W < 1:
        raise ValueError("size dimensions must be positive")
    if H <= max_h:
        return H, W
    scale = max_h / H
    return max_h, max(24, int(round(W * scale)))


def _resize(field: np.ndarray, size: tuple[int, int], order: int = 3) -> np.ndarray:
    H, W = map(int, size)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    field = np.maximum(field, 0.0).astype(np.float32, copy=False)
    if field.shape != (H, W):
        field = ndimage.zoom(
            field, (H / field.shape[0], W / field.shape[1]), order=order,
            mode="wrap", prefilter=order > 1,
        )
        field = field[:H, :W]
        if field.shape != (H, W):
            field = np.pad(field, ((0, H - field.shape[0]),
                                   (0, W - field.shape[1])), mode="edge")
    return np.ascontiguousarray(np.maximum(field, 0), dtype=np.float32)


def _laplace(a: np.ndarray) -> np.ndarray:
    """Isotropic nine-point Laplacian used by the finite-difference solvers."""
    cardinal = (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
                np.roll(a, 1, 1) + np.roll(a, -1, 1))
    diagonal = (np.roll(np.roll(a, 1, 0), 1, 1) +
                np.roll(np.roll(a, 1, 0), -1, 1) +
                np.roll(np.roll(a, -1, 0), 1, 1) +
                np.roll(np.roll(a, -1, 0), -1, 1))
    return 0.2 * cardinal + 0.05 * diagonal - a


def _seed_discs(h, w, rng, spacing=22):
    """A full-canvas constellation of soft reaction nuclei."""
    yy, xx = np.mgrid[:h, :w]
    v = np.zeros((h, w), dtype=np.float32)
    sy = max(10, min(spacing, h // 3))
    sx = max(10, min(spacing, w // 3))
    for cy in range(sy // 2, h, sy):
        for cx in range(sx // 2, w, sx):
            jy = (cy + rng.integers(-sy // 4, sy // 4 + 1)) % h
            jx = (cx + rng.integers(-sx // 4, sx // 4 + 1)) % w
            rad = rng.uniform(2.3, 4.8)
            # Periodic distance keeps nuclei at the frame edges continuous.
            dy = np.minimum(np.abs(yy - jy), h - np.abs(yy - jy))
            dx = np.minimum(np.abs(xx - jx), w - np.abs(xx - jx))
            v = np.maximum(v, np.exp(-(dx * dx + dy * dy) / (2 * rad * rad)))
    return v.astype(np.float32)


def _gray_scott_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    v = 0.74 * _seed_discs(h, w, rng, spacing=max(14, min(h, w) // 5))
    v += 0.018 * rng.random((h, w), dtype=np.float32)
    u = np.clip(1.0 - 0.62 * v, 0.0, 1.0).astype(np.float32)
    memory = v.copy()

    def step(u, v, feed, kill):
        uvv = u * v * v
        un = u + 0.16 * _laplace(u) - uvv + feed * (1.0 - u)
        vn = v + 0.08 * _laplace(v) + uvv - (feed + kill) * v
        return (np.clip(un, 0.0, 1.2).astype(np.float32),
                np.clip(vn, 0.0, 1.2).astype(np.float32))

    for _ in range(90):
        u, v = step(u, v, 0.0350, 0.0600)
    for i in range(n_frames):
        # A closed parameter breath changes spot size without frame-to-frame
        # flicker; the state remains fully deterministic.
        phase = 2 * np.pi * i / max(n_frames, 1)
        feed = 0.0350 + 0.0014 * np.sin(phase)
        kill = 0.0600 + 0.0009 * np.sin(phase + 1.2)
        for _ in range(4):
            u, v = step(u, v, feed, kill)
        memory = np.maximum(v, memory * np.float32(0.955))
        edge = ndimage.gaussian_gradient_magnitude(v, 0.72, mode="wrap")
        field = 1.55 * v + 0.75 * memory + 2.8 * edge
        yield _resize(field, size)


def _fitzhugh_frames(n_frames, size, seed, max_h, turing=False):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    if turing:
        # Schnakenberg's activator/inhibitor kinetics are the canonical Turing
        # companion to the excitable FitzHugh--Nagumo medium below.
        a0, b0 = 0.12, 0.88
        u = (a0 + b0) + 0.025 * rng.standard_normal((h, w))
        v = b0 / (a0 + b0) ** 2 + 0.025 * rng.standard_normal((h, w))
        u, v = u.astype(np.float32), v.astype(np.float32)

        def turing_step(u, v):
            uvv = u * u * v
            dt = 0.012
            un = u + dt * (0.8 * _laplace(u) + a0 - u + uvv)
            vn = v + dt * (15.0 * _laplace(v) + b0 - uvv)
            return (np.clip(un, 0.0, 2.5).astype(np.float32),
                    np.clip(vn, 0.0, 2.5).astype(np.float32))

        for _ in range(1100):
            u, v = turing_step(u, v)
        memory = u.copy()
        for _ in range(n_frames):
            for _ in range(14):
                u, v = turing_step(u, v)
            pattern = np.maximum(u - np.percentile(u, 18), 0.0)
            memory = np.maximum(pattern, memory * np.float32(0.97))
            edge = ndimage.gaussian_gradient_magnitude(u, 0.75, mode="wrap")
            yield _resize(1.2 * pattern + 0.55 * memory + 2.6 * edge, size)
        return

    # FitzHugh--Nagumo rest state, disturbed by many broken wave fronts.  Their
    # collisions generate persistent excitable scrolls across the tall canvas.
    u = np.full((h, w), -1.2, dtype=np.float32)
    v = np.full((h, w), -0.625, dtype=np.float32)
    yy, xx = np.mgrid[:h, :w]
    tile = max(18, min(h, w) // 3)
    local_x = (xx + tile // 3) % tile
    local_y = (yy + tile // 5) % tile
    excited = (local_x < tile * 0.44) & (local_y < tile * 0.78)
    u[excited] = 1.25
    v[(local_y < tile * 0.36) & excited] = 0.35
    u += 0.018 * rng.standard_normal((h, w)).astype(np.float32)
    memory = np.zeros((h, w), dtype=np.float32)
    pacer = _seed_discs(h, w, rng, spacing=max(16, min(h, w) // 6))
    pacer_group = ((xx // tile) + 2 * (yy // tile)) % 4

    def fhn_step(u, v):
        dt = 0.105
        du = 0.82 * _laplace(u) + u - (u * u * u) / 3.0 - v
        dv = 0.09 * (u + 0.7 - 0.8 * v) + 0.025 * _laplace(v)
        return ((u + dt * du).astype(np.float32),
                (v + dt * dv).astype(np.float32))

    pacer_mask = pacer > 0.72

    def apply_pacemakers(state, frame_index):
        drive = (0.5 + 0.5 * np.sin(2 * np.pi * frame_index / 52.0
                                   + pacer_group * (np.pi / 2))) ** 10
        target = -1.2 + 2.55 * drive
        state[pacer_mask] = np.maximum(state[pacer_mask], target[pacer_mask])

    # Pre-roll the same driven dynamics used during the reel.  The opening
    # frame therefore contains mature rings, not the rectangular seed mask.
    for warm_i in range(-96, 0):
        apply_pacemakers(u, warm_i)
        for _ in range(3):
            u, v = fhn_step(u, v)
    memory = np.clip((u + 1.15) / 2.25, 0.0, 1.0).astype(np.float32)
    for i in range(n_frames):
        # Excitable media need an energy source once the initial fronts have
        # passed.  Staggered smooth pacemakers are a standard FHN boundary
        # condition and keep new rings crossing the whole portrait.
        apply_pacemakers(u, i)
        for _ in range(3):
            u, v = fhn_step(u, v)
        active = np.clip((u + 1.15) / 2.25, 0.0, 1.0)
        memory = np.maximum(active, memory * np.float32(0.94))
        edge = ndimage.gaussian_gradient_magnitude(active, 0.75, mode="wrap")
        field = 1.35 * active + 0.72 * memory + 2.4 * edge
        yield _resize(field, size)


def _bz_frames(n_frames, size, seed, max_h):
    """Barkley excitable kinetics, a visual model of BZ spiral waves."""
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    # A vertical excitation front crossed by refractory bands is the classic
    # numerical way to make free wave tips.  Every tip curls into a counter-
    # rotating BZ spiral, giving several large rotors down the portrait.
    band = max(28, h // 2)
    u = (xx < w * 0.50).astype(np.float32)
    v = ((yy % band) < band * 0.50).astype(np.float32) * 0.72
    u += 0.012 * rng.random((h, w), dtype=np.float32)
    memory = u.copy()

    def step(u, v):
        dt = 0.012
        threshold = (v + 0.055) / 0.75
        reaction = u * (1.0 - u) * (u - threshold) / 0.022
        # The weighted stencil represents roughly 0.3 of a unit-grid
        # Laplacian; this scale gives the BZ fronts room to travel instead of
        # pinning to their initial pixels.
        un = u + dt * (reaction + 8.0 * _laplace(u))
        vn = v + dt * (u - v + 0.10 * _laplace(v))
        return (np.clip(un, 0.0, 1.05).astype(np.float32),
                np.clip(vn, 0.0, 1.05).astype(np.float32))

    for _ in range(560):
        u, v = step(u, v)
    memory = u.copy()
    for _ in range(n_frames):
        for _ in range(7):
            u, v = step(u, v)
        memory = np.maximum(u, memory * np.float32(0.945))
        front = ndimage.gaussian_gradient_magnitude(u, 0.82, mode="wrap")
        field = 1.4 * u + 0.68 * memory + 3.0 * front + 0.22 * v
        yield _resize(field, size)


def _swift_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    u = (0.085 * rng.standard_normal((h, w))).astype(np.float32)
    # Preferred wavelength of ~13 logical pixels: broad enough to upscale into
    # elegant cymatic labyrinths, dense enough to fill a portrait canvas.
    ky = 2 * np.pi * np.fft.fftfreq(h)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(w)[None, :]
    k2 = ky * ky + kx * kx
    q0 = 2 * np.pi / 13.0
    memory = np.abs(u)

    def step(state, control):
        dt = 0.65
        linear = control - (q0 * q0 - k2) ** 2
        nonlinear_hat = np.fft.rfft2(state * state * state)
        new_hat = (np.fft.rfft2(state) - dt * nonlinear_hat) / (1.0 - dt * linear)
        return np.fft.irfft2(new_hat, s=state.shape).real.astype(np.float32)

    for _ in range(150):
        u = step(u, 0.22)
    for i in range(n_frames):
        phase = 2 * np.pi * i / max(n_frames, 1)
        control = 0.22 + 0.025 * np.sin(phase)
        for _ in range(2):
            u = step(u, control)
        pattern = np.abs(u)
        memory = np.maximum(pattern, memory * np.float32(0.965))
        ridges = ndimage.gaussian_gradient_magnitude(u, 0.72, mode="wrap")
        field = 1.15 * pattern + 0.65 * memory + 2.1 * ridges
        yield _resize(field, size)


def frames(kind="gray_scott", n_frames=300, size=(1920, 1080), seed=41,
           max_h=240):
    """Yield exactly ``n_frames`` float32 scalar fields for a PDE model.

    ``fitzhugh_nagumo`` is an excitable medium; ``turing`` selects the related
    stationary activator/inhibitor regime.  The remaining kinds are
    ``gray_scott``, ``belousov_zhabotinsky`` and ``swift_hohenberg``.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown reaction model {kind!r}; choose from {sorted(_KINDS)}")
    if n_frames < 0:
        raise ValueError("n_frames must be nonnegative")
    if kind == "gray_scott":
        yield from _gray_scott_frames(n_frames, size, seed, max_h)
    elif kind == "fitzhugh_nagumo":
        yield from _fitzhugh_frames(n_frames, size, seed, max_h, turing=False)
    elif kind == "turing":
        yield from _fitzhugh_frames(n_frames, size, seed, max_h, turing=True)
    elif kind == "belousov_zhabotinsky":
        yield from _bz_frames(n_frames, size, seed, max_h)
    else:
        yield from _swift_frames(n_frames, size, seed, max_h)
