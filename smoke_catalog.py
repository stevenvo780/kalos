#!/usr/bin/env python3
"""Validate and smoke-render the scene registry on a tiny internal grid.

Examples::

    python3 smoke_catalog.py                     # every registered scene
    python3 smoke_catalog.py lorenz julia-morph  # selected scenes
    python3 smoke_catalog.py --jobs 8            # bounded parallelism
    python3 smoke_catalog.py --one lorenz        # isolated worker mode

The catalog driver launches one subprocess per scene.  Detailed renderer output
is kept in ``out/_smoke/logs`` while the terminal receives one concise result
line per scene.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SMOKE_DIR = ROOT / "out" / "_smoke"
LOG_DIR = SMOKE_DIR / "logs"
TINY_GRID = (96, 54)  # (height, width), preserving the reel's 9:16 aspect
SECONDS = 2.0
FPS = 12


def _load_render() -> Any:
    """Import the live registry and redirect its global grid before use."""
    module = importlib.import_module("render")
    module.GRID = TINY_GRID
    return module


def _safe_scene_name(name: str) -> str:
    """Keep registry names from escaping their smoke output directories."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"scene name is not path-safe: {name!r}")
    return name


def _validate_fields(name: str, spec: Any) -> None:
    """Require exactly two useful fields from the registry generator."""
    try:
        iterator = iter(spec["gen"](2))
    except Exception as exc:
        raise RuntimeError(f"could not start generator: {exc}") from exc

    fields: list[np.ndarray] = []
    for index in range(2):
        try:
            field = next(iterator)
        except StopIteration as exc:
            raise AssertionError(
                f"generator stopped after {index} field(s), expected 2"
            ) from exc
        if not isinstance(field, np.ndarray):
            raise TypeError(
                f"field {index} is {type(field).__name__}, expected ndarray"
            )
        if field.shape != TINY_GRID:
            raise AssertionError(
                f"field {index} shape {field.shape}, expected {TINY_GRID}"
            )
        if field.dtype != np.dtype("float32"):
            raise TypeError(
                f"field {index} dtype {field.dtype}, expected float32"
            )
        if not np.isfinite(field).all():
            raise ValueError(f"field {index} contains NaN or infinity")
        minimum = float(np.min(field))
        maximum = float(np.max(field))
        if not np.any(field):
            raise ValueError(f"field {index} is blank")
        if maximum == minimum:
            raise ValueError(
                f"field {index} is constant ({minimum:.6g}); expected structure"
            )
        fields.append(field)
        print(
            f"validate {name} field={index} min={minimum:.6g} "
            f"max={maximum:.6g} nonzero={np.count_nonzero(field)}/{field.size}"
        )

    try:
        extra = next(iterator)
    except StopIteration:
        extra = None
    else:
        del extra
        raise AssertionError("generator yielded more than the requested 2 fields")

    # Keep both arrays live until validation completes; this catches generators
    # that illegally resize and reuse one backing array between yields.
    if any(field.shape != TINY_GRID for field in fields):
        raise AssertionError("a yielded field changed shape after iteration")


def _run_one(name: str) -> Path:
    """Validate one scene and produce its two-second smoke reel."""
    name = _safe_scene_name(name)
    render = _load_render()
    if name not in render.SCENES:
        raise KeyError(f"unknown scene {name!r}")

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    spec = render.SCENES[name]
    _validate_fields(name, spec)

    destination = SMOKE_DIR / f"{name}.mp4"
    destination.unlink(missing_ok=True)
    # Reel normally removes this directory.  Clear remnants from an interrupted
    # earlier smoke so ffmpeg cannot append stale numbered PNGs.
    frame_dir = SMOKE_DIR / f".frames_{name}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    rendered = Path(render.render_scene(
        name, seconds=SECONDS, fps=FPS, out=str(destination)
    ))
    if not rendered.is_file() or rendered.stat().st_size == 0:
        raise RuntimeError(f"renderer did not create a non-empty {destination}")
    print(f"artifact {rendered} bytes={rendered.stat().st_size}")
    return rendered


def _one_cli(name: str) -> int:
    """Subprocess entry point; its full output is redirected to a scene log."""
    started = time.monotonic()
    try:
        path = _run_one(name)
    except Exception as exc:
        traceback.print_exc()
        print(
            f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"PASS {name}: {path} ({time.monotonic() - started:.1f}s)",
        flush=True,
    )
    return 0


def _subprocess_smoke(name: str) -> tuple[str, int, float, Path]:
    """Run one isolated worker, returning only its concise catalog metadata."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{_safe_scene_name(name)}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--one", name]
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        code = completed.returncode
    except Exception:
        code = 1
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    return name, code, time.monotonic() - started, log_path


def _selected_scenes(positional: list[str]) -> tuple[list[str], list[str]]:
    render = _load_render()
    available = list(render.SCENES)
    requested = positional or available
    # Preserve registry/CLI order and avoid rendering a duplicate twice.
    selected = list(dict.fromkeys(requested))
    unknown = [name for name in selected if name not in render.SCENES]
    return selected, unknown


def _catalog_cli(scene_names: list[str], jobs: int) -> int:
    if jobs < 1:
        print("FAIL catalog: --jobs must be at least 1", file=sys.stderr)
        return 2
    try:
        selected, unknown = _selected_scenes(scene_names)
    except Exception as exc:
        print(
            f"FAIL catalog: could not import render.SCENES: {exc}",
            file=sys.stderr,
        )
        return 2

    if not selected:
        print("FAIL catalog: render.SCENES is empty", file=sys.stderr)
        return 2
    if unknown:
        known = set(selected) - set(unknown)
        for name in selected:
            if name in unknown:
                print(f"FAIL {name}: unknown scene", flush=True)
        if not known:
            return 2
        selected = [name for name in selected if name in known]

    failures = len(unknown)
    worker_count = min(jobs, len(selected))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_subprocess_smoke, name): name for name in selected
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, code, elapsed, log_path = future.result()
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: worker error: {exc}", flush=True)
                continue
            relative_log = log_path.relative_to(ROOT)
            if code == 0:
                print(
                    f"PASS {name} ({elapsed:.1f}s) log={relative_log}",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"FAIL {name} ({elapsed:.1f}s) log={relative_log}",
                    flush=True,
                )
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and smoke-render scenes from render.SCENES"
    )
    parser.add_argument(
        "scenes", nargs="*", metavar="SCENE",
        help="scene names (default: every registered scene)",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="maximum parallel smoke subprocesses (default: 4)",
    )
    parser.add_argument(
        "--one", metavar="NAME",
        help="validate/render one scene in the current process (worker mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.one is not None:
        if args.scenes:
            parser.error("--one cannot be combined with positional scenes")
        return _one_cli(args.one)
    return _catalog_cli(args.scenes, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
