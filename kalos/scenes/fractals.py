"""Animated escape-time fractals as portrait-native scalar fields.

The public :func:`frames` dispatcher mirrors the other kalos scene modules: it
yields exactly ``n_frames`` finite, non-negative float32 arrays with shape
``size``.  Colour is deliberately left to :mod:`kalos.grade`.

All animation paths are closed.  We sample ``t = i / n_frames`` (rather than
duplicating the first frame at the end), so a reel joins cleanly when it loops.
"""
from __future__ import annotations

import numpy as np


KINDS = (
    "mandelbrot",
    "julia",
    "burning_ship",
    "newton",
    "lyapunov",
    "phoenix",
    "multibrot",
)


def _plane(size, center=0j, half_height=1.8, rotation=0.0):
    """Return a complex64 plane with square pixels in a portrait viewport."""
    h, w = map(int, size)
    if h <= 0 or w <= 0:
        raise ValueError("size must contain two positive integers")
    half_width = float(half_height) * w / h
    x = np.linspace(-half_width, half_width, w, dtype=np.float32)
    # Mathematical +imaginary points upward on screen.
    y = np.linspace(half_height, -half_height, h, dtype=np.float32)
    z = (x[None, :] + 1j * y[:, None]).astype(np.complex64)
    if rotation:
        z *= np.complex64(np.exp(1j * rotation))
    z += np.complex64(center)
    return z


def _clean(field):
    """Enforce the scene contract without an avoidable float64 copy."""
    field = np.asarray(field, dtype=np.float32)
    np.nan_to_num(field, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(field, np.float32(0.0), out=field)
    return field


def _smooth_value(iteration, magnitude, degree=2.0):
    """Continuous escape count; callers only pass magnitudes above 2."""
    magnitude = np.maximum(magnitude, np.float32(2.000001))
    degree = max(float(degree), 1.000001)
    value = (iteration + 1.0
             - np.log(np.log(magnitude)) / np.log(np.float32(degree)))
    return np.maximum(value, np.float32(0.0)).astype(np.float32, copy=False)


def _quadratic_escape(c, z, max_iter, *, burning=False, degree=2.0,
                      blend_cubic=0.0, previous_weight=None):
    """Smooth escape field for the quadratic-family scenes.

    ``blend_cubic`` continuously interpolates z**2 and z**3 for Multibrot.
    ``previous_weight`` enables the two-step Phoenix recurrence.
    """
    z = np.asarray(z, dtype=np.complex64).copy()
    c = np.asarray(c, dtype=np.complex64)
    active = np.ones(z.shape, dtype=bool)
    result = np.zeros(z.shape, dtype=np.float32)
    previous = np.zeros_like(z) if previous_weight is not None else None
    bailout2 = np.float32(64.0 if (blend_cubic or previous is not None) else 16.0)
    blend = np.float32(blend_cubic)
    pweight = np.complex64(previous_weight or 0.0)

    for k in range(int(max_iter)):
        if burning:
            work = (np.abs(z.real) + 1j * np.abs(z.imag)).astype(
                np.complex64, copy=False)
        else:
            work = z

        z2 = work * work
        if blend:
            # A polynomial homotopy is much cheaper than a per-pixel complex
            # real power, and moves continuously from the 2- to 3-Multibrot.
            new = z2 * (np.float32(1.0) + blend * (work - 1.0)) + c
        elif previous is not None:
            new = z2 + c + pweight * previous
        else:
            new = z2 + c

        mag2 = new.real * new.real + new.imag * new.imag
        escaped = active & (mag2 > bailout2)
        if np.any(escaped):
            mag = np.sqrt(mag2[escaped]).astype(np.float32, copy=False)
            result[escaped] = _smooth_value(k, mag, degree)

        # Only live orbits advance.  Clearing newly escaped values prevents
        # overflow in later vectorised polynomial evaluations.
        if previous is not None:
            previous[active] = z[active]
        z[active] = new[active]
        active[escaped] = False
        z[escaped] = 0.0j
        if previous is not None:
            previous[escaped] = 0.0j
        if not np.any(active):
            break

    return _clean(result)


def _mandelbrot_frames(n_frames, size, max_iter=72, **_):
    # Seahorse Valley: the breathing zoom travels through dense, recognisable
    # Mandelbrot filaments instead of shrinking the whole set into the frame.
    target = np.complex64(-0.7436439 + 0.1318259j)
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        half_height = 0.38 * np.exp(-1.25 * (1.0 - np.cos(phase)))
        c = _plane(size, target, half_height)
        z = np.zeros_like(c)
        yield _quadratic_escape(c, z, max_iter)


def _julia_frames(n_frames, size, max_iter=62, **_):
    # This small closed parameter orbit stays close to the connected Julia-set
    # boundary, changing dendrites into islands without popping between states.
    base = np.complex64(-0.745 + 0.105j)
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        c = base + np.complex64(0.075 * np.cos(phase)
                                + 0.055j * np.sin(phase))
        # The Julia set's long axis is naturally horizontal; a quarter-turn
        # makes its dendrites tower through the 9:16 canvas.
        rotation = 0.5 * np.pi + 0.10 * np.sin(phase)
        z = _plane(size, 0j, 1.75, rotation)
        yield _quadratic_escape(c, z, max_iter)


def _burning_ship_frames(n_frames, size, max_iter=68, **_):
    # Fly from the complete ship toward its western antenna.  A quarter-turn
    # converts the set's horizontal coastline into a full-height fiery ridge.
    wide_center = np.complex64(-1.00 - 0.45j)
    detail_center = np.complex64(-1.40 - 0.28j)
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        travel = 0.5 - 0.5 * np.cos(phase)
        center = wide_center * (1.0 - travel) + detail_center * travel
        half_height = 1.65 * (1.0 - travel) + 1.05 * travel
        rotation = 0.5 * np.pi + 0.055 * np.sin(phase)
        c = _plane(size, center, half_height, rotation)
        z = np.zeros_like(c)
        yield _quadratic_escape(c, z, max_iter, burning=True)


def _newton_frames(n_frames, size, max_iter=24, **_):
    roots = np.array((1.0 + 0.0j,
                      -0.5 + 0.8660254j,
                      -0.5 - 0.8660254j), dtype=np.complex64)
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        half_height = 2.05 * (1.0 + 0.055 * np.cos(phase))
        z = _plane(size, 0j, half_height, rotation=phase).copy()
        first_convergence = np.full(z.shape, max_iter, dtype=np.float32)
        active = np.ones(z.shape, dtype=bool)

        for k in range(int(max_iter)):
            z2 = z * z
            denominator = 3.0 * z2
            tiny = (denominator.real * denominator.real
                    + denominator.imag * denominator.imag) < 1e-12
            denominator[tiny] = np.complex64(1e-6 + 0.0j)
            new = z - (z2 * z - 1.0) / denominator
            residual = np.abs(new * new * new - 1.0)
            converged = active & (residual < 2e-4)
            first_convergence[converged] = np.float32(k + 1)
            z[active] = new[active]
            active[converged] = False
            if not np.any(active):
                break

        # Root identity itself is categorical, but changes between neighbours
        # reveal the infinitely nested basin boundary as a scalar light field.
        distances = np.stack([np.abs(z - root) for root in roots], axis=0)
        basin = np.argmin(distances, axis=0).astype(np.int8)
        edge = np.zeros(z.shape, dtype=bool)
        edge[:, 1:] |= basin[:, 1:] != basin[:, :-1]
        edge[:, :-1] |= basin[:, 1:] != basin[:, :-1]
        edge[1:, :] |= basin[1:, :] != basin[:-1, :]
        edge[:-1, :] |= basin[1:, :] != basin[:-1, :]
        slow = np.square(first_convergence / np.float32(max_iter))
        field = slow + np.float32(2.75) * edge
        yield _clean(field)


def _lyapunov_frames(n_frames, size, warmup=16, iterations=40,
                     sequence="AABAB", **_):
    """Markus--Lyapunov logistic map, with a closed parameter-plane drift."""
    h, w = map(int, size)
    if not sequence or any(letter not in "AB" for letter in sequence.upper()):
        raise ValueError("Lyapunov sequence must contain only A and B")
    sequence = sequence.upper()

    base_a = np.linspace(2.64, 3.95, w, dtype=np.float32)[None, :]
    base_b = np.linspace(3.96, 2.36, h, dtype=np.float32)[:, None]
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        # Drifting both axes on a tiny ellipse animates the tongues while the
        # broad, useful logistic interval remains full-bleed.
        a = base_a + np.float32(0.035 * np.cos(phase))
        b = base_b + np.float32(0.030 * np.sin(phase))
        x = np.full((h, w), 0.5, dtype=np.float32)
        total = np.zeros((h, w), dtype=np.float32)
        count = 0
        steps = int(warmup) + int(iterations)
        for k in range(steps):
            r = a if sequence[k % len(sequence)] == "A" else b
            x *= r * (1.0 - x)
            # Roundoff can put x a hair outside its invariant interval.
            np.clip(x, 0.0, 1.0, out=x)
            if k >= warmup:
                derivative = np.maximum(np.abs(r * (1.0 - 2.0 * x)), 1e-7)
                total += np.log(derivative)
                count += 1
        exponent = total / np.float32(max(count, 1))
        # Stable periodic tongues (negative exponent) glow over chaotic voids.
        field = np.maximum(-exponent, 0.0)
        yield _clean(field)


def _phoenix_frames(n_frames, size, max_iter=58, **_):
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        # This compact orbit remains in the connected, bird-like regime.  A
        # wider sweep crosses a bifurcation and washes the whole field out.
        c = np.complex64(0.552 + 0.012 * np.cos(phase)
                         + 0.012j * np.sin(phase))
        previous_weight = np.float32(-0.52 + 0.012 * np.sin(phase + 0.7))
        z = _plane(size, 0j, 1.65, rotation=0.06 * np.sin(phase))
        yield _quadratic_escape(c, z, max_iter,
                                previous_weight=previous_weight)


def _multibrot_frames(n_frames, size, max_iter=56, **_):
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        # 0 -> 1 -> 0 is a closed homotopy between z^2+c and z^3+c.
        blend = np.float32(0.5 - 0.5 * np.cos(phase))
        center = np.complex64((-0.47 + 0.10 * blend)
                              + 0.025j * np.sin(phase))
        c = _plane(size, center, 1.48)
        z = np.zeros_like(c)
        yield _quadratic_escape(c, z, max_iter,
                                degree=2.0 + float(blend),
                                blend_cubic=blend)


_GENERATORS = {
    "mandelbrot": _mandelbrot_frames,
    "julia": _julia_frames,
    "burning_ship": _burning_ship_frames,
    "newton": _newton_frames,
    "lyapunov": _lyapunov_frames,
    "phoenix": _phoenix_frames,
    "multibrot": _multibrot_frames,
}

_ALIASES = {
    "mandelbrot_zoom": "mandelbrot",
    "julia_morph": "julia",
    "newton_fractal": "newton",
    "lyapunov_markus": "lyapunov",
    "markus": "lyapunov",
    "phoenix_fractal": "phoenix",
    "multibrot_morph": "multibrot",
}


def frames(kind="mandelbrot", n_frames=300, size=(1920, 1080), **kwargs):
    """Yield an animated fractal field.

    ``kind`` is one of :data:`KINDS`; common descriptive names such as
    ``"mandelbrot_zoom"`` and hyphenated spellings are accepted as aliases.
    Scene-specific controls (for example ``max_iter``) pass through ``kwargs``.
    """
    n_frames = int(n_frames)
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    try:
        generator = _GENERATORS[key]
    except KeyError as exc:
        raise ValueError(f"unknown fractal '{kind}'; have {', '.join(KINDS)}") from exc
    yield from generator(n_frames, size, **kwargs)
