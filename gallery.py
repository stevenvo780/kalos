#!/usr/bin/env python3
"""Render the whole catalog to out/ — one reel per scene. Systematic mode."""
from __future__ import annotations
import sys
from render import SCENES, render_scene


def main(seconds: float = 12, fps: int = 30):
    for name in SCENES:
        try:
            render_scene(name, seconds, fps)
        except Exception as e:  # keep going; one bad scene shouldn't stop the batch
            print(f"FAIL {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(s)
