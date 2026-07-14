"""A gallery of fifteen strange attractors rendered as density fields.

The continuous systems are integrated as deterministic vectorized ensembles and
the maps as batches of nearby orbits.  A slow, closed camera path turns their
point clouds into coherent animation without re-rolling noise every frame.

All public output is a finite, non-negative ``float32`` array.  The robust,
axis-independent camera fit is intentional: it lets each attractor occupy the
entire portrait canvas instead of becoming a small diagram in its centre.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


TAU = np.float32(2.0 * np.pi)

KINDS = (
    "lorenz",
    "aizawa",
    "rossler",
    "thomas",
    "halvorsen",
    "dadras",
    "chen",
    "gumowski_mira",
    "hopalong",
    "ikeda",
    "tinkerbell",
    "svensson",
    "bedhead",
    "duffing",
    "standard_map",
)


# ---------------------------------------------------------------------------
# Continuous systems


def _lorenz(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    return np.stack((10.0 * (y - x), x * (28.0 - z) - y,
                     x * y - (8.0 / 3.0) * z))


def _aizawa(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    r2 = x * x + y * y
    return np.stack(((z - 0.7) * x - 3.5 * y,
                     3.5 * x + (z - 0.7) * y,
                     0.6 + 0.95 * z - z * z * z / 3.0
                     - r2 * (1.0 + 0.25 * z) + 0.1 * z * x * x * x))


def _rossler(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    return np.stack((-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)))


def _thomas(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    b = 0.208186
    return np.stack((np.sin(y) - b * x, np.sin(z) - b * y,
                     np.sin(x) - b * z))


def _halvorsen(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    a = 1.4
    return np.stack((-a * x - 4.0 * y - 4.0 * z - y * y,
                     -a * y - 4.0 * z - 4.0 * x - z * z,
                     -a * z - 4.0 * x - 4.0 * y - x * x))


def _dadras(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    return np.stack((y - 3.0 * x + 2.7 * y * z,
                     1.7 * y - x * z + z,
                     2.0 * x * y - 9.0 * z))


def _chen(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    return np.stack((35.0 * (y - x), -7.0 * x - x * z + 28.0 * y,
                     x * y - 3.0 * z))


def _duffing(s: np.ndarray) -> np.ndarray:
    x, velocity, phase = s
    return np.stack((velocity,
                     -0.2 * velocity + x - x * x * x
                     + 0.34 * np.cos(phase),
                     np.full_like(x, 1.2)))


_ODE_CONFIG: dict[str, dict[str, object]] = {
    "lorenz": dict(rhs=_lorenz, centre=(0.0, 0.0, 24.0),
                   spread=(13.0, 13.0, 11.0), dt=0.006, burn=2600,
                   stride=4, limit=90.0, view=(0.05, 0.30, 0.12)),
    "aizawa": dict(rhs=_aizawa, centre=(0.1, 0.0, 0.0),
                   spread=(0.7, 0.7, 0.55), dt=0.004, burn=2400,
                   stride=5, limit=8.0, view=(0.40, 0.44, -0.18)),
    "rossler": dict(rhs=_rossler, centre=(0.0, 0.0, 0.5),
                    spread=(2.0, 2.0, 0.35), dt=0.010, burn=2300,
                    stride=6, limit=50.0, view=(-0.35, 0.36, 0.16)),
    "thomas": dict(rhs=_thomas, centre=(0.0, 0.0, 0.0),
                   spread=(2.1, 2.1, 2.1), dt=0.035, burn=1500,
                   stride=4, limit=18.0, view=(0.30, 0.48, 0.22)),
    "halvorsen": dict(rhs=_halvorsen, centre=(0.2, 0.0, -0.2),
                      spread=(0.8, 0.8, 0.8), dt=0.0035, burn=2800,
                      stride=6, limit=30.0, view=(-0.42, 0.52, -0.20)),
    "dadras": dict(rhs=_dadras, centre=(1.0, 1.0, 1.0),
                   spread=(0.8, 0.8, 0.7), dt=0.004, burn=2500,
                   stride=5, limit=35.0, view=(0.24, 0.35, 0.24)),
    "chen": dict(rhs=_chen, centre=(0.0, 0.0, 20.0),
                 spread=(11.0, 11.0, 9.0), dt=0.0012, burn=3300,
                 stride=7, limit=100.0, view=(-0.10, 0.32, 0.15)),
    "duffing": dict(rhs=_duffing, centre=(0.0, 0.0, 0.0),
                    spread=(1.1, 0.8, 0.0), dt=np.pi / 180.0, burn=3600,
                    stride=300, limit=1000.0, view=(0.05, 0.34, -0.12)),
}


def _reset_bad(state: np.ndarray, bad: np.ndarray, centre: np.ndarray,
               spread: np.ndarray, rng: np.random.Generator) -> None:
    """Deterministically reseed rare numerical escapees without losing a frame."""
    count = int(np.count_nonzero(bad))
    if count:
        state[:, bad] = centre[:, None] + spread[:, None] * rng.normal(
            0.0, 0.45, (state.shape[0], count)).astype("float32")


def _ode_cloud(kind: str, points: int, seed: int) -> np.ndarray:
    cfg = _ODE_CONFIG[kind]
    rhs = cfg["rhs"]
    assert callable(rhs)
    centre = np.asarray(cfg["centre"], dtype="float32")
    spread = np.asarray(cfg["spread"], dtype="float32")
    dt = np.float32(cfg["dt"])
    limit = np.float32(cfg["limit"])
    burn = int(cfg["burn"])
    stride = int(cfg["stride"])

    # Enough independent trajectories to make a smooth density, while keeping
    # the thousands of integration steps cache-friendly.
    width = min(4096, max(768, (points + 47) // 48))
    rng = np.random.default_rng(seed)
    state = centre[:, None] + spread[:, None] * rng.normal(
        0.0, 0.72, (3, width)).astype("float32")
    if kind == "duffing":
        # Equal forcing phase plus one-period sampling below forms a Poincare
        # section, exposing the folded chaotic set instead of filling a box.
        state[2] = 0.0

    samples = (points + width - 1) // width
    chunks: list[np.ndarray] = []
    total_steps = burn + samples * stride
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(total_steps):
            # Explicit midpoint is noticeably cleaner than Euler for Chen and
            # Halvorsen, at only two vector-field evaluations per step.
            k1 = rhs(state)  # type: ignore[operator]
            midpoint = state + np.float32(0.5) * dt * k1
            state += dt * rhs(midpoint)  # type: ignore[operator]
            bad = (~np.all(np.isfinite(state), axis=0)
                   | np.any(np.abs(state) > limit, axis=0))
            _reset_bad(state, bad, centre, spread, rng)
            if step >= burn and (step - burn) % stride == 0:
                chunks.append(state.copy())

    cloud = np.concatenate(chunks, axis=1)[:, :points]
    if kind == "duffing":
        # Lift the 2-D Poincare section into a shallow folded surface so a
        # moving camera can reveal its two lobes without adding fake noise.
        cloud[2] = (cloud[0] * cloud[1]).astype("float32")
    return cloud.astype("float32", copy=False)


# ---------------------------------------------------------------------------
# Discrete maps


MapStep = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _gumowski_mira(x: np.ndarray,
                   y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = -0.918

    def f(q: np.ndarray) -> np.ndarray:
        return mu * q + 2.0 * (1.0 - mu) * q * q / (1.0 + q * q)

    xn = y + 0.008 * (1.0 - 0.05 * y * y) * y + f(x)
    return xn, -x + f(xn)


def _hopalong(x: np.ndarray,
              y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xn = y - np.sign(x) * np.sqrt(np.abs(x))
    return xn, np.float32(0.4) - x


def _ikeda(x: np.ndarray,
           y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 0.4 - 6.0 / (1.0 + x * x + y * y)
    c, s = np.cos(angle), np.sin(angle)
    return 1.0 + 0.9 * (x * c - y * s), 0.9 * (x * s + y * c)


def _tinkerbell(x: np.ndarray,
                y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x * x - y * y + 0.9 * x - 0.6013 * y,
            2.0 * x * y + 2.0 * x + 0.5 * y)


def _svensson(x: np.ndarray,
              y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (-6.56 * np.sin(1.4 * x) - np.sin(1.56 * y),
            1.4 * np.cos(1.4 * x) + np.cos(1.56 * y))


def _bedhead(x: np.ndarray,
             y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = 0.65343, 0.7345345
    return (np.sin(x * y / b) * y + np.cos(a * x - y),
            x + np.sin(y) / b)


def _standard_map(x: np.ndarray,
                  y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x is angle and y angular momentum; both coordinates live on a torus.
    yn = np.remainder(y + 0.971635 * np.sin(x) + np.pi, TAU) - np.pi
    xn = np.remainder(x + yn + np.pi, TAU) - np.pi
    return xn, yn


_MAP_CONFIG: dict[str, dict[str, object]] = {
    "gumowski_mira": dict(step=_gumowski_mira, burn=850, stride=3,
                          limit=45.0, init=(0.15, 0.15), view=(0.12, 0.20)),
    "hopalong": dict(step=_hopalong, burn=40, stride=1,
                     limit=2500.0, init=(0.003, 0.003), view=(-0.12, 0.14)),
    "ikeda": dict(step=_ikeda, burn=180, stride=2,
                  limit=12.0, init=(0.10, 0.10), view=(0.22, 0.28)),
    "tinkerbell": dict(step=_tinkerbell, burn=700, stride=2,
                       limit=8.0, init=(0.10, 0.10), view=(-0.20, 0.22)),
    "svensson": dict(step=_svensson, burn=120, stride=2,
                     limit=20.0, init=(1.2, 1.2), view=(0.04, 0.18)),
    "bedhead": dict(step=_bedhead, burn=300, stride=2,
                    limit=80.0, init=(0.7, 0.7), view=(0.28, 0.22)),
    "standard_map": dict(step=_standard_map, burn=0, stride=1,
                         limit=8.0, init=(np.pi, 0.14), view=(0.0, 0.10)),
}


def _initial_map_points(kind: str, width: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = _MAP_CONFIG[kind]["init"]  # type: ignore[misc]
    if kind == "standard_map":
        # Thin phase-space ribbons reveal both KAM islands and the chaotic sea;
        # a uniform torus would merely look like television noise.
        x = rng.uniform(-np.pi, np.pi, width).astype("float32")
        bands = rng.choice(np.array([-1.55, -0.42, 0.0, 0.42, 1.55],
                                    dtype="float32"), width)
        y = bands + rng.normal(0.0, float(sy), width).astype("float32")
        return x, y.astype("float32")
    if kind == "hopalong":
        # Nearby seeds generate nested, offset Hopalong wings.
        x = rng.normal(0.0, float(sx), width).astype("float32")
        y = rng.normal(0.0, float(sy), width).astype("float32")
        return x, y
    return (rng.normal(0.0, float(sx), width).astype("float32"),
            rng.normal(0.0, float(sy), width).astype("float32"))


def _map_cloud(kind: str, points: int, seed: int) -> np.ndarray:
    cfg = _MAP_CONFIG[kind]
    step_fn = cfg["step"]
    assert callable(step_fn)
    burn = int(cfg["burn"])
    stride = int(cfg["stride"])
    limit = np.float32(cfg["limit"])
    # Hopalong is orbit art: a few very long trajectories preserve its nested
    # wings, whereas thousands of short offset paths wash them into an ellipse.
    width = (32 if kind == "hopalong"
             else min(8192, max(1024, (points + 47) // 48)))
    rng = np.random.default_rng(seed)
    x, y = _initial_map_points(kind, width, rng)
    samples = (points + width - 1) // width
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for iteration in range(burn + samples * stride):
            x, y = step_fn(x, y)  # type: ignore[operator]
            x = np.asarray(x, dtype="float32")
            y = np.asarray(y, dtype="float32")
            bad = (~np.isfinite(x) | ~np.isfinite(y)
                   | (np.abs(x) > limit) | (np.abs(y) > limit))
            count = int(np.count_nonzero(bad))
            if count:
                xr, yr = _initial_map_points(kind, count, rng)
                x[bad], y[bad] = xr, yr
            if iteration >= burn and (iteration - burn) % stride == 0:
                xs.append(x.copy())
                ys.append(y.copy())

    return np.stack((np.concatenate(xs)[:points],
                     np.concatenate(ys)[:points])).astype("float32")


# ---------------------------------------------------------------------------
# Portrait projection and public generator


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    centre = np.median(values, axis=1, keepdims=True)
    shifted = values - centre
    scale = np.percentile(np.abs(shifted), 99.65, axis=1, keepdims=True)
    return (shifted / np.maximum(scale, 1e-5)).astype("float32")


def _histogram(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
               size: tuple[int, int]) -> np.ndarray:
    h, w = size
    # A little clipping makes rare numerical outliers harmless and ensures the
    # useful structure reaches all four sides of the portrait.
    u = np.clip(u, -1.04, 1.04)
    v = np.clip(v, -1.04, 1.04)
    dn = (depth - np.min(depth)) / (np.ptp(depth) + 1e-6)
    weights = (0.58 + 0.72 * dn).astype("float32")
    field, _, _ = np.histogram2d(
        v, u, bins=(h, w), range=((-1.05, 1.05), (-1.05, 1.05)),
        weights=weights)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(field, 0.0).astype("float32", copy=False)


def _ode_frames(kind: str, cloud: np.ndarray, n_frames: int,
                size: tuple[int, int]) -> Iterator[np.ndarray]:
    p = _robust_standardize(cloud)
    yaw0, yaw_amp, pitch_amp = _ODE_CONFIG[kind]["view"]  # type: ignore[misc]
    for index in range(n_frames):
        phase = 2.0 * np.pi * index / n_frames
        yaw = float(yaw0) + float(yaw_amp) * np.sin(phase)
        pitch = (0.13 * np.cos(phase)
                 + float(pitch_amp) * np.sin(phase + 0.65))
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        side = cy * p[0] + sy * p[1]
        depth = -sy * p[0] + cy * p[1]
        vertical = cp * p[2] + sp * depth
        depth2 = -sp * p[2] + cp * depth

        # Closed breathing/shear motion adds life without changing the model.
        side = side + np.float32(0.055 * np.sin(phase + 1.1)) * vertical
        projected = _robust_standardize(np.stack((side, vertical)))
        yield _histogram(projected[0] * 0.985, projected[1] * 0.985,
                         depth2, size)


def _map_frames(kind: str, cloud: np.ndarray, n_frames: int,
                size: tuple[int, int]) -> Iterator[np.ndarray]:
    p = _robust_standardize(cloud)
    angle0, angle_amp = _MAP_CONFIG[kind]["view"]  # type: ignore[misc]
    for index in range(n_frames):
        phase = 2.0 * np.pi * index / n_frames
        angle = float(angle0) + float(angle_amp) * np.sin(phase)
        c, s = np.cos(angle), np.sin(angle)
        u = c * p[0] - s * p[1]
        v = s * p[0] + c * p[1]
        # Periodic shear creates a graceful loop and preserves fine map detail.
        u = u + np.float32(0.045 * np.sin(phase + 0.8)) * v
        q = _robust_standardize(np.stack((u, v)))
        depth = (0.55 * np.sin(1.7 * p[0] + phase)
                 + 0.45 * np.cos(1.3 * p[1] - phase))
        yield _histogram(q[0] * 0.985, q[1] * 0.985, depth, size)


def frames(kind: str = "lorenz", n_frames: int = 300,
           size: tuple[int, int] = (1920, 1080), points: int | None = None,
           seed: int = 713) -> Iterator[np.ndarray]:
    """Yield exactly ``n_frames`` portrait density fields for ``kind``.

    ``points=None`` scales point count with the grid, capped for predictable
    full-reel render time.  Hyphenated names are accepted for registry clarity
    (``standard-map`` and ``gumowski-mira``).
    """
    kind = kind.lower().replace("-", "_")
    if kind not in KINDS:
        raise ValueError(f"unknown strange attractor {kind!r}; have {KINDS}")
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("size must be a positive (height, width) pair")
    if n_frames == 0:
        return

    if points is None:
        points = max(70_000, min(280_000, int(size[0] * size[1] * 0.56)))
    if points < 1:
        raise ValueError("points must be positive")

    if kind in _ODE_CONFIG:
        cloud = _ode_cloud(kind, int(points), seed)
        yield from _ode_frames(kind, cloud, n_frames, size)
    else:
        cloud = _map_cloud(kind, int(points), seed)
        yield from _map_frames(kind, cloud, n_frames, size)
