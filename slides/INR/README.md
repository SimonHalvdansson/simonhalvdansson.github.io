# Implicit Neural Representations — a 20-minute talk

An HTML slide deck with live, in-browser INR training (TensorFlow.js on WebGPU).

## Presenting

Open `index.html` in Chrome or Edge (double-click works, no server needed), press **F** for fullscreen
and page with the arrow keys or a clicker. Everything is local, so it also works offline.

| Key | Action |
| --- | --- |
| → ↓ PgDn Space | next build step / slide |
| ← ↑ PgUp | previous |
| F (or a clicker's F5) | fullscreen |
| R | restart the current slide (re-runs its animation or training) |
| P | presenter window with notes, next slide and a timer |
| O / Esc | slide overview |
| B or . | blank screen |
| digits + Enter | jump to slide |
| 1 / 2 / 3 | on the live CT slide: 15 / 30 / 90 projections |
| W | on the last slide: train an INR on the webcam, live (asks for camera access) |

The URL hash (`#/5.1` = slide 5, build step 1) is kept in sync, so reloading stays on the same slide.

Live demos (trained in the browser when the slide opens): the chirp (slides 6 and 9), the image with
Fourier features (slide 10) and sparse-view CT (slide 14). The first time TensorFlow.js runs, WebGPU
compiles its shaders; the deck does that in the background while the title slide is up, so open the
deck a minute before you start. Without WebGPU the demos fall back to WebGL (slower); the image demo
falls back to a precomputed replay if neither is available. The plain-ReLU comparison on slide 5 is
precomputed (5000 steps per network would take too long live).

## Slides

1. Title
2. Universal approximation theorem (ReLU hinges summing to a fit, N = 3 → 48)
3. Functions are more than curves: audio, image, video, signed distance, radiance field, wind field
4. Implicit neural representation: pixel grid vs. f_θ(x, y) = (r, g, b)
5. Attempt 1: ReLU MLPs from 1k to 791k parameters (precomputed, log-loss curves)
6. The chirp with two ReLU MLPs (132k and 790k) — live, with spectrum (spectral bias)
7. Fourier series reminder: partial sums + epicycles
8. Fourier features (Tancik et al. 2020) with a 1989 → today timeline
9. The chirp with Fourier features — live, deliberately paced
10. Attempt 2: Fourier features on the image — live
11. Why INRs are cool: continuous, differentiable, plug into physics, compact
12. CT: rotating gantry building a sinogram
13. Solving the inverse problem: min ‖A(f_θ) − y‖² as a pipeline
14. Sparse-view CT, live: INR vs. filtered back-projection
15. Other problems (CT, NeRF, PINNs, phase retrieval, super-resolution)
16. Neural texture compression (NVIDIA): 9-channel material, relit render, storage
17. Takeaways
18. That's it! (press W for a live webcam INR)

## Files

```
index.html          slides (static content, formulas, speaker notes in <aside class="notes">)
css/                deck + per-slide styles
js/deck.js          slide engine: scaling stage, build steps, keys, presenter view
js/inr.js           TensorFlow.js INR models and trainers (image, 1D signal, CT), shader warm-up
js/slides/*.js      one module per slide (animations, live demos)
js/plot.js, mlpsvg.js, gl.js, util.js   small helpers (plots, MLP diagrams, WebGL)
assets/*.js         precomputed data as base64 (so file:// works): ReLU runs, images, material maps
vendor/             TensorFlow.js 4.22 (from Every Frame an INR), KaTeX 0.16
fonts/              Fira Sans / Fira Mono
tools/              Python scripts that generate assets/ (PyTorch), plus a no-cache dev server
```

Regenerate the precomputed assets (PyTorch, uses MPS on Apple silicon):

```bash
python3 tools/make_relu_runs.py
```

```bash
python3 tools/make_media.py
```

While editing, `python3 tools/serve.py` serves the folder at http://127.0.0.1:8791 without caching.

## Credits

- TensorFlow.js (Apache-2.0); the training loop follows *Every Frame an INR*.
- KaTeX (MIT) for Computer Modern formulas; Fira Sans / Fira Mono (SIL OFL).
- Photos and the tram clip come from the *Every Frame an INR* repository (Unsplash); see `tools/sources/CREDITS.md`.
- The brick material is procedural (`tools/make_media.py`). Neural-texture numbers are from Table 3 of
  Vaidyanathan et al., *Random-Access Neural Compression of Material Textures*, SIGGRAPH 2023.
