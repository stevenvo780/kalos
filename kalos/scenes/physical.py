"""Spreading systems, waves, chaos and statistical-physics scenes."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    label,
    zoom,
)


KINDS = (
    "sir_epidemic", "forest_fire", "percolation", "wave_interference",
    "chladni", "double_pendulum", "nbody", "curl_flow", "phyllotaxis",
    "maurer_rose", "fourier_epicycles", "ising",
)


def _logical_size(size, longest=224):
    h, w = size
    s = min(1.0, longest / max(h, w))
    return max(48, int(round(h * s))), max(36, int(round(w * s)))


def _resize(field, size, order=1):
    h, w = size
    field = np.asarray(field, dtype="float32")
    if field.shape == (h, w):
        return field
    out = zoom(field, (h / field.shape[0], w / field.shape[1]), order=order,
               mode="nearest", prefilter=False)
    fixed = np.empty((h, w), dtype="float32")
    hh, ww = min(h, out.shape[0]), min(w, out.shape[1])
    fixed[:hh, :ww] = out[:hh, :ww]
    if hh < h:
        fixed[hh:] = fixed[hh - 1]
    if ww < w:
        fixed[:, ww:] = fixed[:, ww - 1:ww]
    return fixed


def _neighbors(mask):
    mask = np.asarray(mask, dtype="uint8")
    return (np.roll(mask, 1, 0) + np.roll(mask, -1, 0) +
            np.roll(mask, 1, 1) + np.roll(mask, -1, 1) +
            np.roll(np.roll(mask, 1, 0), 1, 1) +
            np.roll(np.roll(mask, 1, 0), -1, 1) +
            np.roll(np.roll(mask, -1, 0), 1, 1) +
            np.roll(np.roll(mask, -1, 0), -1, 1))


def _sir(n_frames, size, seed=613):
    """Spatial SIRS lattice: recurrent epidemic fronts with immune wakes."""
    lh, lw = _logical_size(size, 236)
    rng = np.random.default_rng(seed)
    state = np.zeros((lh, lw), dtype="uint8")  # susceptible/infected/recovered
    age = np.zeros((lh, lw), dtype="int16")
    yy, xx = np.mgrid[:lh, :lw]
    for sy, sx in ((.12, .22), (.24, .78), (.49, .47), (.72, .16), (.84, .76)):
        seed_disc = (yy - sy * lh) ** 2 + (xx - sx * lw) ** 2 < 2.8 ** 2
        state[seed_disc] = 1
    trail = np.zeros((lh, lw), dtype="float32")
    for frame in range(n_frames):
        for _ in range(2):
            infected = state == 1
            count = _neighbors(infected)
            new = (state == 0) & (count > 0) & (rng.random(state.shape) < 1 - .71 ** count)
            age[state != 0] += 1
            recover = infected & (age >= 8)
            lose_immunity = (state == 2) & (age >= 64)
            state[new] = 1; age[new] = 0
            state[recover] = 2; age[recover] = 0
            state[lose_immunity] = 0; age[lose_immunity] = 0
            if np.count_nonzero(state == 1) < max(3, lh * lw // 2800):
                # Imported cases keep the spatial model active on long reels.
                picks = rng.integers(0, lh * lw, max(2, lh * lw // 5000))
                py, px = np.unravel_index(picks, state.shape)
                state[py, px] = 1; age[py, px] = 0
        hot = (state == 1).astype("float32")
        trail = np.maximum(trail * .91, hot)
        immune_edge = (state == 2) & (_neighbors(state == 0) > 0)
        field = gaussian_filter(hot * 3.4 + trail * 1.4 + immune_edge * .55, .72,
                                mode="wrap")
        yield _resize(field, size)


def _forest_fire(n_frames, size, seed=991):
    """Drossel--Schwabl forest fire with branching flame fronts."""
    lh, lw = _logical_size(size, 232)
    rng = np.random.default_rng(seed)
    woodland = gaussian_filter(rng.random((lh, lw)), 1.35, mode="wrap")
    trees = woodland > np.quantile(woodland, .22)
    fertility = gaussian_filter(rng.random((lh, lw)), 2.4, mode="wrap")
    fertile = fertility > np.quantile(fertility, .42)
    burning = np.zeros((lh, lw), dtype=bool)
    ash_age = np.zeros((lh, lw), dtype="int16")
    # A few initial lightning strikes give frame one an intentional composition.
    for y, x in ((.08, .18), (.28, .74), (.55, .42), (.78, .82), (.91, .20)):
        burning[int(y * lh), int(x * lw)] = True
    trees[burning] = False
    embers = np.zeros((lh, lw), dtype="float32")
    for frame in range(n_frames):
        for _ in range(2):
            ignite = trees & (_neighbors(burning) > 0)
            lightning = trees & (rng.random(trees.shape) < 0.00012)
            old_fire = burning
            empty = (~trees) & (~burning)
            ash_age[empty] = np.minimum(ash_age[empty] + 1, 30000)
            ash_age[old_fire] = 1
            grow_edge = empty & (_neighbors(trees) >= 2) & (ash_age > 16)
            grow_core = empty & fertile & (ash_age > 42)
            grow = ((grow_edge & (rng.random(trees.shape) < .040)) |
                    (grow_core & (rng.random(trees.shape) < .055)) |
                    (empty & (ash_age > 92)))
            burning = ignite | lightning
            trees = (trees | grow) & ~burning & ~old_fire
            ash_age[trees] = 0
            if burning.sum() < 3:
                candidates = np.flatnonzero(trees)
                if len(candidates):
                    chosen = rng.choice(candidates, min(4, len(candidates)), replace=False)
                    burning.flat[chosen] = True; trees.flat[chosen] = False
        embers = np.maximum(embers * .90, burning.astype("float32"))
        heat = gaussian_filter(embers, 2.0, mode="wrap")
        field = gaussian_filter(burning * 4.0 + embers * 2.1, .65, mode="wrap")
        field += heat * .65
        yield _resize(field, size)


def _percolation(n_frames, size, seed=204):
    """Correlated site percolation with cluster mass mapped to luminosity."""
    lh, lw = _logical_size(size, 260)
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((lh, lw)), 1.45, mode="wrap")
    b = gaussian_filter(rng.standard_normal((lh, lw)), 3.8, mode="wrap")
    noise = a + b * .72
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        p = .535 + .075 * np.sin(ph)
        threshold = np.quantile(noise, p)
        occupied = noise < threshold
        labs, count = label(occupied)
        masses = np.bincount(labs.ravel())
        masses[0] = 0
        mapped = np.log1p(masses[labs]).astype("float32")
        if mapped.max() > 0:
            mapped /= mapped.max()
        edge = occupied ^ binary_erosion(occupied)
        field = gaussian_filter(mapped * .65 + edge * (1.1 + mapped), .62)
        yield _resize(field, size)


def _waves(n_frames, size):
    lh, lw = _logical_size(size, 300)
    aspect = lh / lw
    y, x = np.mgrid[-aspect:aspect:complex(lh), -1:1:complex(lw)].astype("float32")
    sources = ((-.62, -1.18, 1.0, 8.0), (.58, -.72, .82, 9.4),
               (-.35, .05, .92, 11.2), (.48, .66, .76, 7.3),
               (-.08, 1.31, .68, 12.5))
    radii = [np.hypot(x - sx, y - sy) for sx, sy, _, _ in sources]
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        wave = np.zeros_like(x)
        for j, ((_, _, amp, k), r) in enumerate(zip(sources, radii)):
            wave += amp * np.cos(k * r - ph * (2 + j % 3)) / np.sqrt(.18 + r)
        # Energy density reveals nodal lace as well as incandescent crests.
        field = np.square(wave) + .24 * np.square(np.gradient(wave, axis=0))
        yield _resize(field, size)


def _chladni(n_frames, size):
    lh, lw = _logical_size(size, 320)
    y, x = np.mgrid[-1:1:complex(lh), -1:1:complex(lw)].astype("float32")
    modes = ((3, 5), (4, 7), (5, 8))
    patterns = []
    for m, n in modes:
        patterns.append(np.cos(m * np.pi * x) * np.cos(n * np.pi * y) -
                        np.cos(n * np.pi * x) * np.cos(m * np.pi * y))
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        z = patterns[0] * np.cos(ph) + patterns[1] * np.sin(ph)
        z += .32 * patterns[2] * np.sin(2 * ph + .6)
        sand = np.exp(-np.abs(z) * 16.0)
        fine = np.exp(-np.abs(z) * 45.0)
        field = sand * .55 + fine * 1.8
        yield _resize(field.astype("float32"), size)


def _pendulum_accel(a, b, av, bv, g=9.81):
    delta = a - b
    den = 3.0 - np.cos(2.0 * delta)
    aa = (-3.0 * g * np.sin(a) - g * np.sin(a - 2.0 * b)
          - 2.0 * np.sin(delta) * (bv * bv + av * av * np.cos(delta))) / den
    ba = (2.0 * np.sin(delta) *
          (2.0 * av * av + 2.0 * g * np.cos(a) + bv * bv * np.cos(delta))) / den
    return aa, ba


def _pendulum_rk4(a, b, av, bv, dt):
    """One stable vectorized RK4 step for an equal-mass double pendulum."""
    def derivative(qa, qb, va, vb):
        aa, ba = _pendulum_accel(qa, qb, va, vb)
        return va, vb, aa, ba

    k1 = derivative(a, b, av, bv)
    k2 = derivative(a + .5 * dt * k1[0], b + .5 * dt * k1[1],
                    av + .5 * dt * k1[2], bv + .5 * dt * k1[3])
    k3 = derivative(a + .5 * dt * k2[0], b + .5 * dt * k2[1],
                    av + .5 * dt * k2[2], bv + .5 * dt * k2[3])
    k4 = derivative(a + dt * k3[0], b + dt * k3[1],
                    av + dt * k3[2], bv + dt * k3[3])
    sixth = dt / 6.0
    return tuple((q + sixth * (p1 + 2 * p2 + 2 * p3 + p4)).astype("float32")
                 for q, p1, p2, p3, p4 in zip((a, b, av, bv), k1, k2, k3, k4))


def _double_pendulum(n_frames, size, seed=88):
    lh, lw = _logical_size(size, 300)
    n = int(np.clip(lh * lw * .10, 2_400, 4_800))
    # A one-dimensional ribbon of nearby initial conditions remains a folded
    # line under Hamiltonian flow.  Its repeated stretching is the chaos fan;
    # a random 2-D cloud would only paint a washed-out oval.
    q = np.linspace(-1.0, 1.0, n, dtype="float32")
    a = (2.25 + .72 * q + .025 * np.sin(31 * q)).astype("float32")
    b = (2.62 - .58 * q + .020 * np.sin(23 * q)).astype("float32")
    av = (.015 * np.sin(17 * q)).astype("float32")
    bv = (.015 * np.cos(19 * q)).astype("float32")
    trail = np.zeros((lh, lw), dtype="float32")
    dt = .005
    # Start after the ribbon has folded into a frame-filling fan; the opening
    # frame doubles as the catalog poster and must not be a tiny initial arc.
    for _ in range(400):
        a, b, av, bv = _pendulum_rk4(a, b, av, bv, dt)
    # Integrate only the visually rich pre-mixing interval, then retrace it in
    # reverse.  Hamiltonian mechanics is time-reversible, and this triangular
    # playback closes the reel without periodic resets or a late hazy disk.
    forward_count = 1 if n_frames <= 1 else n_frames // 2 + 1
    logical = []
    for frame in range(forward_count):
        deposit = np.zeros((lh, lw), dtype="float32")
        for _ in range(3):
            a, b, av, bv = _pendulum_rk4(a, b, av, bv, dt)
            a = (a + np.pi) % (2 * np.pi) - np.pi
            b = (b + np.pi) % (2 * np.pi) - np.pi
            x1, y1 = np.sin(a), np.cos(a)
            x2, y2 = x1 + np.sin(b), y1 + np.cos(b)
            px = ((x2 / 2.35 + .5) * lw).astype("int32")
            py = ((y2 / 4.05 + .5) * lh).astype("int32")
            good = (px >= 0) & (px < lw) & (py >= 0) & (py < lh)
            ids = py[good] * lw + px[good]
            deposit += np.bincount(ids, minlength=lh * lw).reshape(lh, lw)
            # The first bob adds the upper ribs from which the second-bob fan hangs.
            px1 = ((x1 / 2.35 + .5) * lw).astype("int32")
            py1 = (((y1 - 1.0) / 4.05 + .5) * lh).astype("int32")
            good1 = (px1 >= 0) & (px1 < lw) & (py1 >= 0) & (py1 < lh)
            ids1 = py1[good1] * lw + px1[good1]
            deposit += .22 * np.bincount(ids1, minlength=lh * lw).reshape(lh, lw)
        trail = trail * .86 + gaussian_filter(deposit, .48) * .13
        logical.append((trail + gaussian_filter(deposit, 1.05) * .18).astype("float32"))
    indices = list(range(forward_count))
    remaining = n_frames - len(indices)
    indices.extend(range(forward_count - 2, forward_count - 2 - remaining, -1))
    for index in indices:
        yield _resize(logical[index], size)


def _particle_mesh_force(x, y, lh, lw):
    ix = np.floor(x).astype("int32") % lw
    iy = np.floor(y).astype("int32") % lh
    rho = np.bincount(iy * lw + ix, minlength=lh * lw).reshape(lh, lw).astype("float32")
    rho -= rho.mean()
    rhok = np.fft.rfft2(rho)
    ky = (2 * np.pi * np.fft.fftfreq(lh))[:, None]
    kx = (2 * np.pi * np.fft.rfftfreq(lw))[None, :]
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    ax = np.fft.irfft2(1j * kx * rhok / k2, s=(lh, lw)).real.astype("float32")
    ay = np.fft.irfft2(1j * ky * rhok / k2, s=(lh, lw)).real.astype("float32")
    return ax, ay, rho + rho.mean()


def _nbody(n_frames, size, seed=144):
    """Periodic particle-mesh N-body gravity: merging miniature galaxies."""
    lh, lw = _logical_size(size, 224)
    rng = np.random.default_rng(seed)
    n = int(np.clip(lh * lw * .46, 4_500, 13_000))
    centers = np.array(((.22, .15), (.72, .28), (.38, .54), (.79, .72), (.18, .84)))
    group = rng.integers(0, len(centers), n)
    x = (centers[group, 0] * lw + rng.normal(0, lw * .105, n)) % lw
    y = (centers[group, 1] * lh + rng.normal(0, lh * .075, n)) % lh
    dx = x - centers[group, 0] * lw
    dy = y - centers[group, 1] * lh
    spin = np.where(group % 2, -1.0, 1.0)
    vx = (-dy * .006 * spin + rng.normal(0, .025, n)).astype("float32")
    vy = (dx * .006 * spin + rng.normal(0, .025, n)).astype("float32")
    x = x.astype("float32"); y = y.astype("float32")
    trail = np.zeros((lh, lw), dtype="float32")
    for frame in range(n_frames):
        ax, ay, _ = _particle_mesh_force(x, y, lh, lw)
        ix = x.astype("int32") % lw; iy = y.astype("int32") % lh
        vx = (vx + ax[iy, ix] * .0008) * .999
        vy = (vy + ay[iy, ix] * .0008) * .999
        x = (x + vx * .18) % lw; y = (y + vy * .18) % lh
        ix = np.floor(x).astype("int32") % lw
        iy = np.floor(y).astype("int32") % lh
        ids = iy * lw + ix
        dens = np.bincount(ids, minlength=lh * lw).reshape(lh, lw).astype("float32")
        trail = trail * .91 + gaussian_filter(dens, .68, mode="wrap") * .30
        yield _resize(trail + gaussian_filter(dens, 1.8, mode="wrap") * .5, size)


def _curl_flow(n_frames, size, seed=377):
    """Particles advected by the curl of a smoothly interpolated noise field."""
    lh, lw = _logical_size(size, 240)
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((lh, lw)), 7.0, mode="wrap")
    b = gaussian_filter(rng.standard_normal((lh, lw)), 7.0, mode="wrap")
    n = int(np.clip(lh * lw * .38, 4_000, 13_000))
    x = rng.uniform(0, lw, n).astype("float32")
    y = rng.uniform(0, lh, n).astype("float32")
    trail = np.zeros((lh, lw), dtype="float32")
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        potential = a * np.cos(ph) + b * np.sin(ph)
        gy, gx = np.gradient(potential)
        ix = x.astype("int32") % lw; iy = y.astype("int32") % lh
        ux, uy = gy[iy, ix], -gx[iy, ix]
        norm = np.hypot(ux, uy) + 1e-6
        x = (x + ux / norm * 1.18) % lw
        y = (y + uy / norm * 1.18) % lh
        ids = y.astype("int32") * lw + x.astype("int32")
        dens = np.bincount(ids, minlength=lh * lw).reshape(lh, lw).astype("float32")
        trail = gaussian_filter(trail * .94 + dens * .13, .62, mode="wrap")
        yield _resize(trail + gaussian_filter(dens, 1.5, mode="wrap") * .32, size)


def _phyllotaxis(n_frames, size):
    h, w = size
    n = int(np.clip(h * w * .115, 20_000, 72_000))
    k = np.arange(n, dtype="float32") + .5
    r = np.sqrt(k / n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        theta = k * golden + ph + .16 * np.sin(ph) * np.sqrt(k / n)
        x = np.cos(theta) * r * .995
        y = np.sin(theta) * r * .995
        # The affine portrait projection deliberately lets the sunflower fill 9:16.
        dens, _, _ = np.histogram2d(y, x, bins=(h, w), range=((-1, 1), (-1, 1)))
        field = gaussian_filter(dens.astype("float32"), .48)
        yield field


def _maurer(n_frames, size):
    h, w = size
    vertices = np.arange(0, 721, dtype="float32")
    u = np.linspace(0, 1, 42, endpoint=False, dtype="float32")
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        d = np.deg2rad(71.0 + 5.5 * np.sin(ph))
        theta = vertices * d
        radius = np.sin(7.0 * theta)
        xv, yv = radius * np.cos(theta), radius * np.sin(theta)
        x = (xv[:-1, None] * (1 - u) + xv[1:, None] * u).ravel()
        y = (yv[:-1, None] * (1 - u) + yv[1:, None] * u).ravel()
        # Two faint counter-rotated copies turn the classic polygon into lace.
        ca, sa = np.cos(.38), np.sin(.38)
        x2, y2 = (x * ca - y * sa) * .91, (x * sa + y * ca) * .91
        xx = np.concatenate((x, x2)); yy = np.concatenate((y, y2))
        dens, _, _ = np.histogram2d(yy, xx, bins=(h, w),
                                    range=((-1.02, 1.02), (-1.02, 1.02)))
        yield gaussian_filter(dens.astype("float32"), .42)


def _heart_fourier(samples=4096):
    u = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    x = 16 * np.sin(u) ** 3
    y = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
    z = x + 1j * y
    coeff = np.fft.fft(z) / samples
    freq = np.fft.fftfreq(samples) * samples
    order = np.argsort(np.abs(coeff))[::-1][:32]
    return freq[order].astype("int32"), coeff[order]


def _fourier(n_frames, size):
    h, w = size
    freq, coeff = _heart_fourier()
    u = np.linspace(0, 2 * np.pi, 18000, endpoint=False, dtype="float32")
    circle_u = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    harmonic_phase = np.arange(len(freq)) * 1.618
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        wobble = .055 * np.sin(ph + harmonic_phase) / np.sqrt(1 + np.abs(freq))
        terms = coeff[:, None] * np.exp(1j * (freq[:, None] * u + wobble[:, None]))
        z = terms.sum(axis=0)
        # Draw the actual epicycle chain at a moving reconstruction parameter.
        cursor = ph
        center = 0j
        circles = []
        links = []
        for f, c, q in zip(freq[:12], coeff[:12], wobble[:12]):
            radius = abs(c)
            circles.append(center + radius * np.exp(1j * circle_u))
            nxt = center + c * np.exp(1j * (f * cursor + q))
            links.append(np.linspace(center, nxt, 80, endpoint=False))
            center = nxt
        ornament = np.concatenate(circles + links)
        zz = np.concatenate((z, ornament))
        dens, _, _ = np.histogram2d(zz.imag, zz.real, bins=(h, w),
                                    range=((-18.5, 13.7), (-18.0, 18.0)))
        yield gaussian_filter(dens.astype("float32"), .48)


def _ising(n_frames, size, seed=812):
    lh, lw = _logical_size(size, 240)
    rng = np.random.default_rng(seed)
    spin = rng.choice(np.array((-1, 1), dtype="int8"), (lh, lw))
    yy, xx = np.mgrid[:lh, :lw]
    parity = (xx + yy) & 1

    def sweep(temp):
        nonlocal spin
        for color in (0, 1):
            nn = (np.roll(spin, 1, 0) + np.roll(spin, -1, 0) +
                  np.roll(spin, 1, 1) + np.roll(spin, -1, 1))
            de = 2 * spin * nn
            flip = (parity == color) & ((de <= 0) | (rng.random(spin.shape) < np.exp(-de / temp)))
            spin[flip] *= -1

    for _ in range(42):
        sweep(1.68)
    trail = np.zeros((lh, lw), dtype="float32")
    for i in range(n_frames):
        ph = 2 * np.pi * i / n_frames
        temp = 1.80 + .32 * np.cos(ph)
        sweep(temp); sweep(temp)
        walls = ((spin != np.roll(spin, 1, 0)) |
                 (spin != np.roll(spin, 1, 1))).astype("float32")
        trail = np.maximum(trail * .86, walls)
        field = gaussian_filter(walls * 1.8 + trail * .75 + (spin > 0) * .10,
                                .58, mode="wrap")
        yield _resize(field, size)


def frames(kind="wave_interference", n_frames=300, size=(1920, 1080), seed=613):
    """Yield one physical model as exactly ``n_frames`` float32 fields."""
    if kind not in KINDS:
        raise ValueError(f"unknown physical model {kind!r}; have {KINDS}")
    table = {
        "sir_epidemic": _sir,
        "forest_fire": _forest_fire,
        "percolation": _percolation,
        "double_pendulum": _double_pendulum,
        "nbody": _nbody,
        "curl_flow": _curl_flow,
        "ising": _ising,
    }
    if kind in table:
        yield from table[kind](n_frames, size, seed)
    elif kind == "wave_interference":
        yield from _waves(n_frames, size)
    elif kind == "chladni":
        yield from _chladni(n_frames, size)
    elif kind == "phyllotaxis":
        yield from _phyllotaxis(n_frames, size)
    elif kind == "maurer_rose":
        yield from _maurer(n_frames, size)
    else:
        yield from _fourier(n_frames, size)
