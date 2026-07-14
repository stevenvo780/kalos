# kalos — expand to 50 mathematical models (autonomous build)

You are a background coding worker on the repo /workspace/kalos (Python, numpy+ffmpeg,
public GitHub github.com/stevenvo780/kalos). It renders mathematical objects as
color-graded 9:16 Instagram reels. It currently has 3 scenes. GOAL: grow the catalog
to **>=50 distinct mathematical models**, each a spectacular reel.

## The scene contract (STUDY the existing code first, match it EXACTLY)
Read: render.py, kalos/video.py, kalos/grade.py, kalos/palettes.py,
kalos/scenes/attractor.py, kalos/scenes/harmonograph.py.
- A scene is a Python generator `frames(n_frames, size=(H,W), ...)` that YIELDS
  n_frames float32 arrays of shape (H,W) — a scalar density/height/intensity field.
- render.py maps each field through grade(field, palette=..., bloom=..., gamma=...) -> RGB,
  and video.py encodes to 1080x1920 mp4. Internal GRID is half-res (upscaled on encode).
- Adding a model = write a generator (new file under kalos/scenes/ or add a function),
  then add ONE entry to the SCENES dict in render.py with a fitting palette + grade recipe.
- Loops must be seamless where the math allows (param sweep base+amp*sin(2*pi*t) so t=0==t=1).

## Quality bar (the owner is picky — this matters)
- Every reel must be VISUALLY SPECTACULAR and FULL-BLEED (fills the 9:16 frame, no
  letterbox, no filler, no tiny centered blob). Legible structure, not TV static.
- Color-graded (palette + bloom/glow + vignette). Deep black voids, incandescent cores.
- If a scene renders blank / washed-out / noisy-mess, TUNE it (params, normalization
  percentile, palette) until it's beautiful, or DROP it. Zero ugly fillers ship.
- Add new palettes to kalos/palettes.py if a model needs one (e.g. viridis-like,
  magma, plasma, deep-sea, toxic-green for slime, coral for growth).

## The 50+ models to implement (grouped; prioritize the ones marked ***)
STRANGE ATTRACTORS (density histograms, cheap+gorgeous): Lorenz, Aizawa, Rossler,
Thomas, Halvorsen, Dadras, Chen, Gumowski-Mira, Hopalong, Ikeda, Tinkerbell,
Svensson, Bedhead, Duffing, standard-map(Chirikov).
FRACTALS (smooth escape-time field, zoom/morph over frames): ***Mandelbrot zoom,
***Julia morph, Burning Ship, Newton fractal, Lyapunov(Markus) fractal, Phoenix,
Multibrot morph.
CELLULAR AUTOMATA / LIFE: ***Conway Game of Life (with decaying trail field),
***Lenia (smooth organic life), SmoothLife, Brian's Brain, cyclic CA (spirals),
elementary CA rule110/rule30 scroll, Langton's ant.
REACTION-DIFFUSION / PATTERNS: ***Gray-Scott (several pattern regimes),
Turing/FitzHugh-Nagumo, ***Belousov-Zhabotinsky spirals, Swift-Hohenberg.
GROWTH / BIOLOGICAL (owner asked for fungi/mycelium): ***Physarum slime mold
(mycelium network — this is the 'hongos' he wants), ***DLA (coral/frost dendrites),
Eden growth, Barnsley fern IFS, L-system plant.
SPREADING / PHYSICAL: ***SIR epidemic on lattice (spreading waves), forest-fire,
percolation cluster, wave interference/ripple, ***Chladni/cymatics plate,
double-pendulum ensemble (chaos fan), N-body gravity, curl/Perlin flow field,
phyllotaxis (sunflower), rose/Maurer curves, Fourier epicycles, Ising domains.
=> That is ~54 candidates. Ship >=50 that pass the quality bar.

## Execution plan — TWO WAVES, checkpoint between
WAVE A (code + prove): implement ALL generators + register each in render.py.
For EACH scene do a fast SMOKE render (2s, 12fps, at a tiny grid) into out/_smoke/
to prove it (a) runs without error and (b) yields a non-blank field. Fix or drop
failures. When all pass: `git add -A && git commit` in logical chunks (by family)
and `git push`. Now the 50-model engine is REAL and public even before full renders.
WAVE B (full catalog): render every kept scene at full quality (12-14s, 30fps) into
out/. Curate: open/inspect, drop any that are ugly (log which + why). Copy the final
mp4s + a poster (first frame png) for each into the owner's showcase:
/datos/workspaces/personal/media-kit/instagram/12-kalos-matematicas/ IF that path is
reachable from here; if not, leave them in out/ and note it. Update README.md with the
full scene list. Commit + push.

## Compute
32 cores, 89GB free RAM available. You MAY render scenes in PARALLEL (e.g. a bash loop
with xargs -P 6) to go faster. Keep each mp4 IG-friendly: 1080x1920, <20MB ideally,
<50MB hard cap (tune crf/points). ffmpeg 6.1 present.

## Discipline (IMPORTANT)
- Keep a running log in _WORKER_STATUS.md at repo root: update it after each family
  completes (what's done, what dropped and why, wave/phase, next step). Overwrite, keep
  it SHORT (<40 lines). This is how the owner tracks you.
- Do NOT print huge output. Work in files. Small numpy sanity checks only.
- Commit in coherent chunks with clear messages. Push to origin/main (or master — check
  `git branch`). Repo is already authed via gh.
- Do not touch anything outside /workspace/kalos except the showcase copy dir in wave B.
- Be autonomous. Do not stop to ask. If a model is too slow/ugly, drop it and note it;
  the target is >=50 beautiful reels, not every candidate.

Start with WAVE A now. Study the existing 3 scenes first so the new ones match the
contract perfectly.
