#!/usr/bin/env python3
"""Render the catalog in bounded parallel subprocesses.

Examples::

    python3 gallery.py                         # all scenes, 12 s / 30 fps
    python3 gallery.py --jobs 8 --force        # rebuild existing reels
    python3 gallery.py fractal-julia life-lenia

Detailed renderer output lives in ``out/_logs``; the terminal receives one
concise status line per scene.  Existing non-empty MP4s are resumed by default.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

from render import SCENES


ROOT = Path(__file__).resolve().parent


def _render_one(name: str, seconds: float, fps: int, out_dir: Path,
                force: bool) -> tuple[str, str, float, Path]:
    destination = out_dir / f"{name}.mp4"
    log_path = out_dir / "_logs" / f"{name}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        return name, "SKIP", 0.0, log_path
    destination.unlink(missing_ok=True)
    started = time.monotonic()
    command = [
        sys.executable, str(ROOT / "render.py"), name,
        str(seconds), str(fps), str(destination),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    status = "PASS" if completed.returncode == 0 and destination.is_file() else "FAIL"
    return name, status, time.monotonic() - started, log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene names (default: all)")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    names = list(dict.fromkeys(args.scenes or SCENES.keys()))
    unknown = [name for name in names if name not in SCENES]
    if unknown:
        parser.error(f"unknown scenes: {', '.join(unknown)}")
    if args.jobs < 1 or args.seconds <= 0 or args.fps < 1:
        parser.error("jobs, seconds and fps must be positive")

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(names))) as pool:
        futures = {
            pool.submit(_render_one, name, args.seconds, args.fps, out_dir,
                        args.force): name
            for name in names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, status, elapsed, log_path = future.result()
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {exc}", flush=True)
                continue
            if status == "FAIL":
                failures += 1
            relative_log = log_path.relative_to(ROOT)
            print(f"{status} {name} ({elapsed:.1f}s) log={relative_log}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
