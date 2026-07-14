"""Cellular automata rendered as luminous, full-frame scalar fields.

The simulations run on a compact toroidal lattice and are enlarged only when a
frame is emitted.  This keeps long reels inexpensive while preserving coherent
cells, soft trails and enough fine structure for the bloom stage.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


_KINDS = {
    "conway", "lenia", "smoothlife", "brians_brain", "cyclic_ca",
    "rule110", "rule30", "langtons_ant",
}


def _logical_shape(size: tuple[int, int], max_h: int = 240) -> tuple[int, int]:
    H, W = map(int, size)
    if H < 1 or W < 1:
        raise ValueError("size dimensions must be positive")
    if H <= max_h:
        return H, W
    scale = max_h / H
    return max_h, max(24, int(round(W * scale)))


def _resize(field: np.ndarray, size: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize to exactly *size* and enforce the public field contract."""
    H, W = map(int, size)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    field = np.maximum(field, 0.0).astype(np.float32, copy=False)
    if field.shape != (H, W):
        field = ndimage.zoom(
            field, (H / field.shape[0], W / field.shape[1]), order=order,
            mode="wrap", prefilter=order > 1,
        )
        # scipy's rounding is normally exact, but make the contract unconditional.
        field = field[:H, :W]
        if field.shape != (H, W):
            field = np.pad(field, ((0, H - field.shape[0]),
                                   (0, W - field.shape[1])), mode="edge")
    return np.ascontiguousarray(field, dtype=np.float32)


def _neighbors(a: np.ndarray) -> np.ndarray:
    """Eight-neighbour Moore sum with seamless/toroidal edges."""
    return sum(
        np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if dx or dy
    )


def _life_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    # Broad probability waves break up the initial soup into continent-sized
    # active zones, avoiding both white noise and an isolated central object.
    p = 0.245 + 0.055 * np.sin(xx * 2 * np.pi / max(w, 1) * 3.0)
    p += 0.035 * np.cos(yy * 2 * np.pi / max(h, 1) * 5.0)
    alive = rng.random((h, w)) < p

    def step(a):
        n = _neighbors(a.astype(np.uint8))
        return (n == 3) | (a & (n == 2))

    for _ in range(10):
        alive = step(alive)
    trail = alive.astype(np.float32)
    for _ in range(n_frames):
        alive = step(alive)
        trail *= np.float32(0.91)
        trail = np.maximum(trail, alive.astype(np.float32))
        halo = ndimage.gaussian_filter(trail, 0.7, mode="wrap")
        field = 1.75 * alive + trail + 0.5 * halo
        yield _resize(field, size, order=1)


def _fft_kernel(h, w, inner, outer, soft=0.0):
    y = np.minimum(np.arange(h), h - np.arange(h))[:, None]
    x = np.minimum(np.arange(w), w - np.arange(w))[None, :]
    r = np.hypot(y, x)
    if soft:
        kernel = np.exp(-0.5 * ((r - (inner + outer) * 0.5) / soft) ** 2)
        kernel *= r <= outer * 1.3
    else:
        kernel = ((r >= inner) & (r <= outer)).astype(np.float32)
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum() + 1e-12
    return np.fft.rfft2(kernel)


def _lenia_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    R = max(4.0, min(h, w) * 0.075)
    khat = _fft_kernel(h, w, R * 0.18, R, soft=R * 0.17)

    # A band-limited field of overlapping organisms is more reliable and more
    # cinematic than a lone canonical Orbium on a tall canvas.
    raw = rng.random((h, w), dtype=np.float32)
    broad = ndimage.gaussian_filter(raw, max(1.0, R * 0.42), mode="wrap")
    broad = (broad - broad.min()) / (np.ptp(broad) + 1e-6)
    a = np.clip(0.06 + 0.29 * broad, 0.0, 1.0).astype(np.float32)
    memory = a.copy()

    def step(state, mu=0.178):
        u = np.fft.irfft2(np.fft.rfft2(state) * khat, s=state.shape).real
        # Lenia's smooth bell-shaped growth response.  A slightly broad sigma
        # supports many morphologies instead of a fragile single specimen.
        growth = 2.0 * np.exp(-0.5 * ((u - mu) / 0.038) ** 2) - 1.0
        nxt = np.clip(state + 0.075 * growth, 0.0, 1.0)
        return ndimage.gaussian_filter(nxt, 0.32, mode="wrap").astype(np.float32)

    for _ in range(18):
        a = step(a)
    for i in range(n_frames):
        # A closed growth-centre sweep makes settled colonies visibly breathe
        # while preserving the underlying Lenia dynamics.
        mu = 0.178 + 0.009 * np.sin(2 * np.pi * i / max(n_frames, 1))
        a = step(a, mu)
        memory = np.maximum(a, memory * np.float32(0.94))
        edge = ndimage.gaussian_gradient_magnitude(a, 0.75, mode="wrap")
        field = 1.45 * a + 0.8 * memory + 2.2 * edge
        yield _resize(field, size, order=3)


def _smoothlife_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    radius = max(4.0, min(h, w) * 0.065)
    inner_hat = _fft_kernel(h, w, 0.0, radius * 0.48)
    outer_hat = _fft_kernel(h, w, radius * 0.50, radius)
    noise = ndimage.gaussian_filter(
        rng.random((h, w), dtype=np.float32), radius * 0.35, mode="wrap")
    threshold = np.quantile(noise, 0.52)
    a = ndimage.gaussian_filter((noise > threshold).astype(np.float32),
                                0.65, mode="wrap")
    trail = a.copy()

    def logistic(x, center, width):
        z = np.clip(4.0 * (x - center) / width, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def step(state):
        ahat = np.fft.rfft2(state)
        m = np.fft.irfft2(ahat * inner_hat, s=state.shape).real
        n = np.fft.irfft2(ahat * outer_hat, s=state.shape).real
        mix = logistic(m, 0.5, 0.15)
        lo = 0.278 * (1.0 - mix) + 0.267 * mix
        hi = 0.365 * (1.0 - mix) + 0.445 * mix
        target = logistic(n, lo, 0.038) * (1.0 - logistic(n, hi, 0.038))
        return np.clip(state + 0.52 * (target - state), 0.0, 1.0).astype(np.float32)

    for _ in range(8):
        a = step(a)
    for _ in range(n_frames):
        a = step(a)
        trail = np.maximum(a, trail * np.float32(0.925))
        contours = ndimage.gaussian_gradient_magnitude(a, 0.8, mode="wrap")
        field = 1.3 * a + 0.72 * trail + 2.5 * contours
        yield _resize(field, size, order=3)


def _brians_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    state = np.zeros((h, w), dtype=np.uint8)
    state[rng.random((h, w)) < 0.145] = 1
    # A few dying cells decorrelate large synchronous patches.
    state[rng.random((h, w)) < 0.045] = 2
    trail = np.zeros((h, w), dtype=np.float32)

    def step(s):
        firing = s == 1
        n = _neighbors(firing.astype(np.uint8))
        out = np.zeros_like(s)
        out[firing] = 2
        out[(s == 0) & (n == 2)] = 1
        return out

    for _ in range(12):
        state = step(state)
    for _ in range(n_frames):
        state = step(state)
        firing = state == 1
        dying = state == 2
        trail *= np.float32(0.87)
        trail = np.maximum(trail, firing.astype(np.float32))
        halo = ndimage.gaussian_filter(trail, 0.72, mode="wrap")
        field = 2.0 * firing + 0.9 * dying + 0.9 * trail + 0.45 * halo
        yield _resize(field, size, order=1)


def _cyclic_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    levels = 16
    state = rng.integers(0, levels, (h, w), dtype=np.int16)
    activity = np.zeros((h, w), dtype=np.float32)

    def step(s):
        target = (s + 1) % levels
        count = sum(
            np.roll(np.roll(s, dy, axis=0), dx, axis=1) == target
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy
        )
        # The one-neighbour threshold is the classic cyclic-wave regime.  A
        # threshold of two freezes surprisingly quickly on narrow reel grids.
        changed = count >= 1
        return np.where(changed, target, s).astype(np.int16), changed

    for _ in range(55):
        state, _ = step(state)
    for _ in range(n_frames):
        state, changed = step(state)
        activity *= np.float32(0.9)
        activity = np.maximum(activity, changed.astype(np.float32))
        phase = state.astype(np.float32) * (2 * np.pi / levels)
        bands = 0.5 + 0.5 * np.sin(phase)
        borders = ndimage.gaussian_gradient_magnitude(bands, 0.65, mode="wrap")
        field = 0.22 * bands + 1.2 * activity + 1.8 * borders
        yield _resize(field, size, order=1)


def _elementary_frames(rule, n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)

    def advance(row):
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        code = (left << 2) | (row << 1) | right
        return ((rule >> code) & 1).astype(np.uint8)

    row = (rng.random(w) < (0.34 if rule == 110 else 0.18)).astype(np.uint8)
    # A deterministic bright landmark keeps the causal weave legible.
    row[w // 2] = 1
    history = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        row = advance(row)
        history[y] = row
    rows_per_frame = max(1, h // 150)
    for _ in range(n_frames):
        history = np.roll(history, -rows_per_frame, axis=0)
        for j in range(rows_per_frame):
            row = advance(row)
            history[-rows_per_frame + j] = row
        # Newly born cells expose the rule's causal chevrons much more clearly
        # than a half-white binary raster.  Dim ancestry keeps the full logical
        # tapestry present behind those incandescent junctions.
        births = history * (1.0 - np.roll(history, 1, axis=0))
        soft = ndimage.gaussian_filter(births, (0.78, 0.48), mode="wrap")
        field = 0.16 * history + 1.8 * births + 0.62 * soft
        yield _resize(field, size, order=1)


def _ant_frames(n_frames, size, seed, max_h):
    h, w = _logical_shape(size, max_h)
    rng = np.random.default_rng(seed)
    board = np.zeros((h, w), dtype=np.uint8)

    # A distributed ensemble makes the global emergent geometry fill portrait
    # video; every member still follows Langton's original two-colour rule.
    ny = max(4, min(8, h // 24))
    nx = max(3, min(6, w // 24))
    ys = np.linspace(0, h, ny, endpoint=False, dtype=int) + h // (2 * ny)
    xs = np.linspace(0, w, nx, endpoint=False, dtype=int) + w // (2 * nx)
    ay, ax = np.meshgrid(ys, xs, indexing="ij")
    ay, ax = ay.ravel(), ax.ravel()
    ay = (ay + rng.integers(-2, 3, len(ay))) % h
    ax = (ax + rng.integers(-2, 3, len(ax))) % w
    direction = rng.integers(0, 4, len(ay), dtype=np.int8)
    trail = np.zeros((h, w), dtype=np.float32)
    dy = np.array([-1, 0, 1, 0], dtype=np.int16)
    dx = np.array([0, 1, 0, -1], dtype=np.int16)

    def ant_step():
        nonlocal ay, ax, direction
        colour = board[ay, ax].copy()
        direction = (direction + np.where(colour == 0, 1, -1)) % 4
        # xor.at preserves correct parity if two ants visit the same cell.
        np.bitwise_xor.at(board, (ay, ax), np.uint8(1))
        ay = (ay + dy[direction]) % h
        ax = (ax + dx[direction]) % w
        np.maximum.at(trail, (ay, ax), np.float32(1.0))

    for _ in range(max(320, 4 * max(h, w))):
        ant_step()
    steps_per_frame = max(24, int(0.42 * min(h, w)))
    for _ in range(n_frames):
        trail *= np.float32(0.965)
        for _ in range(steps_per_frame):
            ant_step()
        halo = ndimage.gaussian_filter(trail, 0.75, mode="wrap")
        field = 0.62 * board + 1.55 * trail + 0.55 * halo
        yield _resize(field, size, order=1)


def frames(kind="conway", n_frames=300, size=(1920, 1080), seed=23,
           max_h=240):
    """Yield exactly ``n_frames`` nonnegative float32 fields of shape ``size``.

    Supported kinds are ``conway``, ``lenia``, ``smoothlife``,
    ``brians_brain``, ``cyclic_ca``, ``rule110``, ``rule30`` and
    ``langtons_ant``.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown automaton {kind!r}; choose from {sorted(_KINDS)}")
    if n_frames < 0:
        raise ValueError("n_frames must be nonnegative")
    if kind == "conway":
        yield from _life_frames(n_frames, size, seed, max_h)
    elif kind == "lenia":
        yield from _lenia_frames(n_frames, size, seed, max_h)
    elif kind == "smoothlife":
        yield from _smoothlife_frames(n_frames, size, seed, max_h)
    elif kind == "brians_brain":
        yield from _brians_frames(n_frames, size, seed, max_h)
    elif kind == "cyclic_ca":
        yield from _cyclic_frames(n_frames, size, seed, max_h)
    elif kind == "rule110":
        yield from _elementary_frames(110, n_frames, size, seed, max_h)
    elif kind == "rule30":
        yield from _elementary_frames(30, n_frames, size, seed, max_h)
    else:
        yield from _ant_frames(n_frames, size, seed, max_h)
