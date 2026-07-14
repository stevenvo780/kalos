"""Growth systems rendered as luminous biological density fields.

The simulations deliberately run on compact logical lattices.  Their fields
are enlarged only at the scene boundary, which keeps long reels practical
without turning cellular detail into unstructured pixel noise.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, zoom


KINDS = ("physarum", "dla", "eden", "barnsley_fern", "lsystem_plant")


def _logical_size(size, longest=220, shortest=96):
    h, w = size
    scale = min(1.0, longest / max(h, w))
    return max(48, int(round(h * scale))), max(36, int(round(w * scale)))


def _resize(field, size, order=1):
    """Resize a logical field and guarantee the exact public scene shape."""
    h, w = size
    if field.shape == (h, w):
        return field.astype("float32", copy=False)
    out = zoom(field, (h / field.shape[0], w / field.shape[1]), order=order,
               mode="nearest", prefilter=False)
    # scipy's rounding is normally exact; the crop/pad makes the contract hard.
    fixed = np.zeros((h, w), dtype="float32")
    hh, ww = min(h, out.shape[0]), min(w, out.shape[1])
    fixed[:hh, :ww] = out[:hh, :ww]
    if hh < h:
        fixed[hh:] = fixed[hh - 1]
    if ww < w:
        fixed[:, ww:] = fixed[:, ww - 1:ww]
    return fixed


def _physarum(n_frames, size, seed=410):
    """Jones-style particle slime mould with chemo-attractant deposition."""
    lh, lw = _logical_size(size, longest=240, shortest=108)
    rng = np.random.default_rng(seed)
    n = int(np.clip(lh * lw * 0.42, 2_800, 13_000))
    y = rng.uniform(0, lh, n).astype("float32")
    x = rng.uniform(0, lw, n).astype("float32")
    angle = rng.uniform(-np.pi, np.pi, n).astype("float32")
    yy, xx = np.mgrid[:lh, :lw].astype("float32")
    trail = (
        0.02 * np.sin(xx * 2 * np.pi / lw * 3.0) ** 2
        + 0.02 * np.sin(yy * 2 * np.pi / lh * 5.0 + xx / lw * 3.0) ** 2
    ).astype("float32")
    sensor, turn, speed = 3.0, 0.42, 1.05

    def sense(offset):
        a = angle + offset
        sx = ((x + np.cos(a) * sensor).astype("int32")) % lw
        sy = ((y + np.sin(a) * sensor).astype("int32")) % lh
        return trail[sy, sx]

    for frame in range(n_frames):
        # A trickle of exploratory agents is part of particle Physarum models:
        # it prevents one winning vein from consuming the full population and
        # keeps a branching mycelial network alive through a long reel.
        if frame % 2 == 0:
            reseed = max(2, n // 80)
            local = trail[y.astype("int32") % lh, x.astype("int32") % lw]
            # Rehome agents from the most overcrowded veins.  This is the
            # particle model's resource-pressure term, not arbitrary noise.
            explorers = np.argpartition(local, -reseed)[-reseed:]
            x[explorers] = rng.uniform(0, lw, len(explorers))
            y[explorers] = rng.uniform(0, lh, len(explorers))
            angle[explorers] = rng.uniform(-np.pi, np.pi, len(explorers))
        # Two short steps preserve filaments while making the organism visibly flow.
        for _ in range(2):
            left, ahead, right = sense(-0.62), sense(0.0), sense(0.62)
            choose_left = (left > ahead) & (left > right)
            choose_right = (right > ahead) & (right > left)
            indecisive = ~(choose_left | choose_right) & (np.abs(left - right) < 0.01)
            angle += turn * (choose_right.astype("float32") - choose_left.astype("float32"))
            angle[indecisive] += rng.uniform(-0.22, 0.22, indecisive.sum())
            angle += rng.normal(0.0, 0.035, n).astype("float32")
            x = (x + np.cos(angle) * speed) % lw
            y = (y + np.sin(angle) * speed) % lh
            ids = y.astype("int32") * lw + x.astype("int32")
            deposit = np.bincount(ids, minlength=lh * lw).reshape(lh, lw).astype("float32")
            trail = gaussian_filter(trail * 0.90 + deposit * 0.075, 0.50,
                                    mode="wrap").astype("float32")
        # Emphasize the living vein network, retaining a faint nutrient haze.
        field = trail + gaussian_filter(deposit, 1.25, mode="wrap") * 0.42
        yield _resize(field, size)


def _make_dla(seed, n=176):
    """Vectorized lattice DLA; returns the cluster and attachment chronology."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=bool)
    c = n // 2
    grid[c, c] = True
    order = [(c, c)]
    radius = 2.0
    target = int(n * n * 0.115)
    batch = max(256, n * 3)
    theta = rng.uniform(0, 2 * np.pi, batch)
    py = np.rint(c + np.sin(theta) * 5).astype("int32")
    px = np.rint(c + np.cos(theta) * 5).astype("int32")
    stalled = 0
    while len(order) < target and radius < n * 0.49 and stalled < n * n * 8:
        direction = rng.integers(0, 4, batch)
        px += (direction == 0).astype("int32") - (direction == 1).astype("int32")
        py += (direction == 2).astype("int32") - (direction == 3).astype("int32")
        dy, dx = py - c, px - c
        far = (dx * dx + dy * dy > (radius + 10.0) ** 2)
        outside = (px < 2) | (px >= n - 2) | (py < 2) | (py >= n - 2)
        relaunch = far | outside
        if relaunch.any():
            a = rng.uniform(0, 2 * np.pi, relaunch.sum())
            launch = min(radius + 5.0, n * 0.485)
            px[relaunch] = np.rint(c + np.cos(a) * launch).astype("int32")
            py[relaunch] = np.rint(c + np.sin(a) * launch).astype("int32")
        hit = (grid[py - 1, px] | grid[py + 1, px] |
               grid[py, px - 1] | grid[py, px + 1]) & ~grid[py, px]
        if hit.any():
            coords = np.unique(np.column_stack((py[hit], px[hit])), axis=0)
            room = target - len(order)
            coords = coords[:room]
            grid[coords[:, 0], coords[:, 1]] = True
            order.extend(map(tuple, coords.tolist()))
            rr = np.sqrt((coords[:, 0] - c) ** 2 + (coords[:, 1] - c) ** 2)
            radius = max(radius, float(rr.max(initial=radius)))
            a = rng.uniform(0, 2 * np.pi, hit.sum())
            launch = min(radius + 5.0, n * 0.485)
            px[hit] = np.rint(c + np.cos(a) * launch).astype("int32")
            py[hit] = np.rint(c + np.sin(a) * launch).astype("int32")
            stalled = 0
        else:
            stalled += 1
    return grid, np.asarray(order, dtype="int32")


def _dla(n_frames, size, seed=73):
    # Generate in a square metric, then stretch it like a frost-covered portrait.
    n = int(np.clip(max(size) // 4, 112, 184))
    cluster, order = _make_dla(seed, n)
    age = np.zeros_like(cluster, dtype="float32")
    age[order[:, 0], order[:, 1]] = np.linspace(0.05, 1.0, len(order), dtype="float32")
    ys, xs = np.nonzero(cluster)
    margin = max(1, int(0.02 * max(np.ptp(ys), np.ptp(xs))))
    y0, y1 = max(0, ys.min() - margin), min(n, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(n, xs.max() + margin + 1)
    cluster = cluster[y0:y1, x0:x1]
    age = age[y0:y1, x0:x1]
    edge = cluster & ~binary_erosion(cluster)
    base = (gaussian_filter(edge.astype("float32"), 0.48) +
            .15 * gaussian_filter(cluster.astype("float32"), 1.0))
    for i in range(n_frames):
        q = i / max(1, n_frames - 1)
        # The complete dendrite is always legible; a bright crystallization front
        # travels from its heart to its outer needles.
        front = np.exp(-((age - (0.12 + 0.88 * q)) / 0.055) ** 2) * edge
        # Keep the complete frost skeleton luminous throughout; only the cool
        # crystallization pulse travels.  Accumulating every revealed cell
        # would turn the old nucleus into an unrelated white disk.
        field = base * 0.72 + gaussian_filter(front.astype("float32"), 1.35) * 0.78
        yield _resize(field, size)


def _eden(n_frames, size, seed=902):
    """Multi-seed Eden growth with a glowing moving frontier and age rings."""
    lh, lw = _logical_size(size, longest=224, shortest=100)
    rng = np.random.default_rng(seed)
    occupied = np.zeros((lh, lw), dtype=bool)
    age = np.full((lh, lw), -1, dtype="int32")
    # A staggered collection of colonies keeps the portrait active edge-to-edge.
    seeds = [(int(lh * y), int(lw * x)) for y, x in
             ((.08, .22), (.12, .78), (.38, .48), (.66, .18), (.62, .82), (.91, .52))]
    for sy, sx in seeds:
        occupied[sy, sx] = True
        age[sy, sx] = 0
    frontier = binary_dilation(occupied) & ~occupied
    trail = np.zeros((lh, lw), dtype="float32")
    additions = max(1, int(np.ceil(lh * lw * 0.78 / max(1, n_frames))))
    for frame in range(n_frames):
        remaining = additions
        while remaining > 0:
            candidates = np.flatnonzero(frontier)
            if not len(candidates):
                break
            # Batched additions retain Eden's random-boundary rule while the
            # explicit budget makes growth span the complete reel.
            take = min(remaining, max(1, len(candidates) // 18), len(candidates))
            chosen = rng.choice(candidates, take, replace=False)
            cy, cx = np.unravel_index(chosen, occupied.shape)
            occupied[cy, cx] = True
            age[cy, cx] = frame + 1
            frontier = binary_dilation(occupied) & ~occupied
            remaining -= take
        edge = binary_dilation(occupied) ^ occupied
        fresh = (age >= max(0, frame - 5)).astype("float32")
        trail = np.maximum(trail * 0.94, fresh)
        rings = (np.sin(np.maximum(age, 0) * 0.42 - frame * 0.34) ** 8) * occupied
        field = occupied.astype("float32") * 0.08 + edge * 1.25 + trail * 1.8 + rings * 0.5
        yield _resize(gaussian_filter(field, 0.65), size)


def _barnsley_points(samples, seed=311):
    rng = np.random.default_rng(seed)
    x = np.zeros(samples, dtype="float32")
    y = np.zeros(samples, dtype="float32")
    r = rng.random(samples)
    # Iteration is sequential, but done only once for the whole reel.
    px = py = 0.0
    for j in range(samples):
        q = r[j]
        if q < .01:
            px, py = 0.0, .16 * py
        elif q < .86:
            px, py = .85 * px + .04 * py, -.04 * px + .85 * py + 1.6
        elif q < .93:
            px, py = .20 * px - .26 * py, .23 * px + .22 * py + 1.6
        else:
            px, py = -.15 * px + .28 * py, .26 * px + .24 * py + .44
        x[j], y[j] = px, py
    return x[100:], y[100:]


def _barnsley(n_frames, size, seed=311):
    h, w = size
    samples = int(np.clip(h * w * 0.42, 70_000, 260_000))
    x0, y0 = _barnsley_points(samples, seed)
    yr = (-.1, 10.05)
    xr = (-2.78, 2.78)
    for i in range(n_frames):
        t = i / n_frames
        sway = 0.13 * np.sin(2 * np.pi * t)
        breathe = 1.0 + 0.025 * np.sin(2 * np.pi * t + 1.2)
        # Height-dependent wind bends the frond while leaving the stem rooted.
        x = x0 * breathe + sway * (y0 / 10.0) ** 2 * 2.0
        y = y0 * (2.0 - breathe)
        dens, _, _ = np.histogram2d(y, x, bins=(h, w), range=(yr, xr))
        yield dens.astype("float32")


def _lsystem_string(depth=5):
    text = "X"
    for _ in range(depth):
        text = "".join("F+[[X]-X]-F[-FX]+X" if c == "X" else "FF" if c == "F" else c
                       for c in text)
    return text


def _plant_points(text, angle, wind, samples_per_segment=4):
    x = y = 0.0
    heading = np.pi / 2
    stack = []
    xs, ys = [], []
    for symbol in text:
        if symbol == "F":
            nx = x + np.cos(heading)
            ny = y + np.sin(heading)
            u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
            # A quadratic wind shear moves the crown more than the trunk.
            yy = y + (ny - y) * u
            xx = x + (nx - x) * u + wind * (yy * yy) * 0.004
            xs.extend(xx); ys.extend(yy)
            x, y = nx, ny
        elif symbol == "+":
            heading += angle
        elif symbol == "-":
            heading -= angle
        elif symbol == "[":
            stack.append((x, y, heading))
        elif symbol == "]" and stack:
            x, y, heading = stack.pop()
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")


def _lsystem(n_frames, size):
    h, w = size
    text = _lsystem_string(5)
    for i in range(n_frames):
        t = i / n_frames
        ph = 2 * np.pi * t
        x, y = _plant_points(text, np.deg2rad(24.0 + 2.2 * np.sin(ph)),
                             np.sin(ph + .7))
        # Three overlapping plants form a full-bleed luminous forest rather than
        # a small specimen centered on a black canvas.
        xx = np.concatenate((x, x * .82 - 8.5, x * .82 + 8.5))
        yy = np.concatenate((y, y * .86 - 1.0, y * .86 - 1.0))
        xmin, xmax = np.percentile(xx, [.15, 99.85])
        ymax = np.percentile(yy, 99.9)
        margin = (xmax - xmin) * .015
        dens, _, _ = np.histogram2d(yy, xx, bins=(h, w),
                                    range=((-1.5, ymax * 1.01),
                                           (xmin - margin, xmax + margin)))
        yield dens.astype("float32")


def frames(kind="physarum", n_frames=300, size=(1920, 1080), seed=410):
    """Yield a biological growth model as float32 fields of exactly ``size``."""
    if kind not in KINDS:
        raise ValueError(f"unknown biological model {kind!r}; have {KINDS}")
    if kind == "physarum":
        yield from _physarum(n_frames, size, seed)
    elif kind == "dla":
        yield from _dla(n_frames, size, seed)
    elif kind == "eden":
        yield from _eden(n_frames, size, seed)
    elif kind == "barnsley_fern":
        yield from _barnsley(n_frames, size, seed)
    else:
        yield from _lsystem(n_frames, size)
