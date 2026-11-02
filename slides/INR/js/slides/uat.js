/* Universal approximation: a one-hidden-layer ReLU network with N neurons fitted (least squares on
   the output layer, hinges at fixed positions) to a wiggly target. N grows 3 -> 6 -> 12 -> 24 -> 48. */
(function () {
  const NX = 500;
  const xs = Float64Array.from({ length: NX }, (_, i) => i / (NX - 1));
  const target = x => 0.62 * Math.sin(2 * Math.PI * 1.15 * x + 0.3) + 0.27 * Math.sin(2 * Math.PI * 3.3 * x + 1.1) + 0.13 * Math.sin(2 * Math.PI * 8.2 * x);
  const ys = xs.map(target);

  /* least squares fit of c + sum_j a_j relu(x - b_j), b_j = j/N */
  function fit(N) {
    const knots = Array.from({ length: N }, (_, j) => j / N);
    const P = N + 1;
    const A = Array.from({ length: P }, () => new Float64Array(P));
    const rhs = new Float64Array(P);
    const phi = new Float64Array(P);
    for (let i = 0; i < NX; i++) {
      phi[0] = 1;
      for (let j = 0; j < N; j++) phi[j + 1] = Math.max(0, xs[i] - knots[j]);
      for (let a = 0; a < P; a++) { rhs[a] += phi[a] * ys[i]; for (let b = 0; b < P; b++) A[a][b] += phi[a] * phi[b]; }
    }
    for (let a = 0; a < P; a++) A[a][a] += 1e-9;
    // gaussian elimination
    const M = A.map((r, i) => [...r, rhs[i]]);
    for (let c = 0; c < P; c++) {
      let piv = c;
      for (let r = c + 1; r < P; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < P; r++) if (r !== c) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= P; k++) M[r][k] -= f * M[c][k];
      }
    }
    const w = M.map((r, i) => r[P] / r[i]);
    const comps = knots.map((b, j) => xs.map(x => w[j + 1] * Math.max(0, x - b)));
    const approx = xs.map((x, i) => w[0] + comps.reduce((s, c) => s + c[i], 0));
    let err = 0;
    for (let i = 0; i < NX; i++) err = Math.max(err, Math.abs(approx[i] - ys[i]));
    return { N, knots, c: w[0], comps, approx, err };
  }

  const NS = [3, 6, 12, 24, 48];
  const fits = {};
  NS.forEach(n => { fits[n] = fit(n); });

  Deck.register('s-uat', {
    steps: 3,
    init(ctx) {
      this.cv = ctx.$('#uat-plot');
      this.readout = ctx.$('.uat-readout');
      this.plot = new Plot(this.cv, {
        w: 900, h: 540, pad: { l: 70, r: 20, t: 20, b: 50 }, font: 22,
        x: { min: 0, max: 1, ticks: 2, tickValues: [0, 0.5, 1], grid: false, label: '' },
        y: { min: -1.25, max: 1.25, tickValues: [-1, 0, 1], grid: true },
      });
      // network diagram (1 -> N -> 1)
      const svg = ctx.$('#uat-net');
      this.net = new MLPDiagram(svg, { layers: [1, 7, 1], x: 60, y: 10, w: 380, h: 250, r: 16, ellipsis: [1] });
      this.net.label(0, 0, 'x', { size: 24 });
      this.net.label(2, 0, '𝒩', { size: 20, italic: false });
      // small ReLU glyph next to the network
      const ix = 500, iy = 70, iw = 120, ih = 80;
      U.svg('line', { x1: ix, y1: iy + ih, x2: ix + iw, y2: iy + ih, stroke: '#c3c2b7', 'stroke-width': 1.5 }, svg);
      U.svg('line', { x1: ix + iw / 2, y1: iy, x2: ix + iw / 2, y2: iy + ih + 6, stroke: '#c3c2b7', 'stroke-width': 1.5 }, svg);
      U.svg('polyline', { points: `${ix},${iy + ih} ${ix + iw / 2},${iy + ih} ${ix + iw},${iy + 4}`, fill: 'none', stroke: '#eb6834', 'stroke-width': 4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      const rl = U.svg('text', { x: ix + iw / 2, y: iy + ih + 36, 'text-anchor': 'middle', 'font-size': 22, fill: '#4b4f5c', 'font-family': 'KaTeX_Main, serif' }, svg);
      rl.innerHTML = '<tspan font-style="italic">σ</tspan>(<tspan font-style="italic">x</tspan>) = max(0, <tspan font-style="italic">x</tspan>)';
      const rl2 = U.svg('text', { x: ix + iw / 2, y: iy + ih + 66, 'text-anchor': 'middle', 'font-size': 24, fill: '#4b4f5c', 'font-family': 'Fira Sans', 'font-weight': 500 }, svg);
      rl2.textContent = 'ReLU';
      this.nLabel = U.svg('text', { x: 250, y: 302, 'text-anchor': 'middle', 'font-size': 24, fill: '#4b4f5c', 'font-family': 'Fira Sans' }, svg);
      this.nLabel.textContent = 'N hidden neurons';
      this.state = { curve: null, comps: [], compAlpha: 0, fx: 1, kn: [], N: 0 };
    },
    resize() { this.plot.resize(); this.draw(); },
    draw() {
      const p = this.plot, s = this.state;
      p.clear();
      p.drawAxes();
      // target (drawn progressively with s.fx in [0,1])
      const upto = Math.max(2, Math.round(s.fx * NX));
      p.line(xs, ys, { color: '#b9b6ad', width: 7, upto });
      // approximation
      if (s.curve) {
        p.line(xs, s.curve, { color: '#eb6834', width: 4, alpha: s.curveAlpha ?? 1 });
        if (s.kn.length && s.kn.length <= 24 && (s.curveAlpha ?? 1) > 0.5) {
          for (const b of s.kn) {
            const i = Math.min(NX - 1, Math.round(b * (NX - 1)));
            p.dot(xs[i], s.curve[i], { r: s.kn.length > 10 ? 4 : 6, color: '#eb6834' });
          }
        }
      }
    },
    setReadout(f) {
      this.readout.innerHTML = f ? `N = ${f.N} · max error ${f.err.toFixed(2)}` : '';
      this.nLabel.textContent = f ? `N = ${f.N} hidden neurons` : 'N hidden neurons';
    },
    showFit(f) {
      Object.assign(this.state, { curve: f.approx.slice(), kn: f.knots, curveAlpha: 1, compAlpha: 0 });
      this.setReadout(f);
    },
    enter(ctx, step) {
      const s = this.state;
      Object.assign(s, { curve: null, comps: [], compAlpha: 0, kn: [], fx: 1 });
      this.setReadout(null);
      if (step === 0) {
        s.fx = 0;
        ctx.run(async c => {
          await c.wait(300);
          await c.play(1400, t => { s.fx = t; this.draw(); });
          await this.net.pulse(c, { dur: 700, color: '#eb6834' });
        });
      } else if (step === 1) {
        this.showFit(fits[3]);
      } else {
        this.showFit(fits[48]);
      }
      this.draw();
    },
    step(ctx, step, prev) {
      const s = this.state;
      this.sub && this.sub.cancel();
      const c = this.sub = ctx.child();
      if (step === 1 && prev === 0) {
        c.run(async c => {
          const f = fits[3];
          s.fx = 1;
          s.kn = [];
          s.curve = null;
          this.net.pulse(c, { dur: 600, color: '#eb6834' }).catch(() => {});
          // grow the sum one neuron at a time: c, then c + h1, c + h1 + h2, ... (a new kink each time)
          let cur = xs.map(() => f.c);
          s.curve = cur.slice(); s.curveAlpha = 0;
          await c.play(400, t => { s.curveAlpha = t; this.draw(); });
          for (let j = 0; j < f.comps.length; j++) {
            const next = cur.map((v, i) => v + f.comps[j][i]);
            const from = cur.slice();
            s.kn = f.knots.slice(0, j + 1);
            await c.play(800, t => { s.curve = from.map((v, i) => v + (next[i] - v) * t); this.draw(); });
            cur = next;
          }
          s.kn = f.knots;
          this.setReadout(f);
          this.draw();
        });
      } else if (step === 2 && prev === 1) {
        c.run(async c => {
          let curF = fits[3];
          for (const n of NS.slice(1)) {
            const f = fits[n], from = s.curve.slice();
            s.kn = [];
            this.net.pulse(c, { dur: 380, color: '#eb6834' }).catch(() => {});
            await c.play(850, t => { s.curve = from.map((v, i) => v + (f.approx[i] - v) * t); this.draw(); });
            s.kn = f.knots; this.setReadout(f); this.draw();
            curF = f;
            await c.wait(650);
          }
        });
      } else if (step < prev) {
        if (step === 0) { Object.assign(s, { curve: null, comps: [], compAlpha: 0, kn: [] }); this.setReadout(null); }
        if (step === 1) this.showFit(fits[3]);
        if (step === 2) this.showFit(fits[48]);
        this.draw();
      }
    },
    leave() { this.sub && this.sub.cancel(); },
  });
})();
