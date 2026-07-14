# kalos catalog worker

- Phase: Wave A — implementation
- Branch: `master` tracking `origin/master`
- Baseline: 3 registered scenes
- Contract reviewed: generator → finite float32 `(H,W)` field → grade → Reel
- Complete: 55 distinct models registered (baseline 3 + 52 new)
- Families: attractors 17, fractals 7, life/CA 8, PDE 5, bio 5, physical 12 + harmonograph
- Numerical smoke: all 55 pass exact count/shape/dtype/finite/nonblank checks
- MP4 smoke: 55/55 pass (2 s × 12 fps, 96×54 grid → 1080×1920)
- Visual pass: all 55 reviewed; 6 weak scenes retuned and now pass final recheck
- Dropped: none
- Next: commit/push Wave A, then full-quality parallel renders + curation
