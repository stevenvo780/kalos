# kalos

**τὸ καλόν — animated mathematical beauty, rendered for Instagram reels.**

`kalos` turns mathematical objects — strange attractors, harmonographs,
fractals, fields — into spectacular 9:16 video loops (1080×1920), with a
color-grading stage (palette + bloom + vignette) that makes the math read like
gallery art instead of a plot.

Every reel is:

- **native 9:16** (1080×1920) — fills the frame, no letterbox, no filler;
- **seamless-looping** where the generator allows it;
- **color-graded** through a curated palette with additive bloom/glow.

## Quick start

```bash
pip install -r requirements.txt
python render.py --list                 # list scenes
python render.py attractor-dejong 12 30 # scene, seconds, fps
python gallery.py                       # render the whole catalog to out/
```

Output lands in `out/<scene>.mp4`.

## Scenes

| scene | math | palette |
|-------|------|---------|
| `attractor-dejong`   | De Jong strange attractor, morphing params | aurora |
| `attractor-clifford` | Clifford strange attractor | neon |
| `harmonograph`       | damped coupled pendulums (Lissajous weave) | ice |

More are added by dropping a frame generator in `kalos/scenes/` and one entry
in `SCENES` (see `render.py`).

## Architecture

```
kalos/
  palettes.py   # curated colormaps (aurora, ember, neon, ice, gold, spectral)
  grade.py      # scalar field -> graded RGB (log-norm + bloom + vignette)
  video.py      # RGB frames -> h264 mp4 (9:16, faststart)
  scenes/       # one module per mathematical object; yields density frames
render.py       # scene registry + single-scene CLI
gallery.py      # batch-render the whole catalog
```

A scene is just a generator that `yield`s 2D `float32` density arrays. `grade`
maps them to color; `video.Reel` encodes them. Pure `numpy` + `scipy` + `ffmpeg` —
no heavyweight animation engine.

## Adding a palette

Edit `kalos/palettes.py` — add `(stop, hex)` control points. The grade stage
accepts any name from `palettes.names()`.

## License

MIT © Steven Vallejo Ortiz — see [LICENSE](LICENSE).
