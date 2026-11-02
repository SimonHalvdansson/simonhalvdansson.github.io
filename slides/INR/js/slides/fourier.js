/* Fourier series reminder: partial sums of a smooth "complicated" periodic signal, and epicycles */
(function () {
  const NX = 600;
  const xs = Float64Array.from({ length: NX }, (_, i) => i / (NX - 1));
  // a smooth periodic "heartbeat-like" signal: periodised gaussian bumps
  const bumps = [[0.18, 0.05, 0.55], [0.36, 0.028, 1.0], [0.42, 0.03, -0.5], [0.50, 0.035, 0.22], [0.70, 0.06, 0.38], [0.86, 0.04, -0.22]];
  const sig = x => {
    let v = -0.15;
    for (const [c, w, a] of bumps) for (const s of [-1, 0, 1]) v += a * Math.exp(-((x - c - s) ** 2) / (2 * w * w));
    return v;
  };
  const ys = xs.map(sig);
  const KMAX = 24;
  // Fourier coefficients by numerical integration
  const M = 4096;
  const A = new Float64Array(KMAX + 1), B = new Float64Array(KMAX + 1);
  for (let m = 0; m < M; m++) {
    const x = m / M, v = sig(x);
    for (let k = 0; k <= KMAX; k++) {
      A[k] += v * Math.cos(2 * Math.PI * k * x) / M * (k === 0 ? 1 : 2);
      B[k] += v * Math.sin(2 * Math.PI * k * x) / M * 2;
    }
  }
  const partial = K => xs.map(x => {
    let v = A[0];
    for (let k = 1; k <= K; k++) v += A[k] * Math.cos(2 * Math.PI * k * x) + B[k] * Math.sin(2 * Math.PI * k * x);
    return v;
  });
  const mags = Array.from({ length: KMAX + 1 }, (_, k) => Math.hypot(A[k], B[k]));
  const KS = [1, 2, 3, 5, 8, 12, 18, 24];

  /* ---- epicycles: a lightning bolt ---- */
  const BOLT = [[0.10, -0.95], [-0.42, 0.10], [-0.04, 0.10], [-0.22, 0.95], [0.42, -0.18], [0.02, -0.18], [0.30, -0.95]];
  function samplePolygon(pts, n) {
    const segs = [];
    let total = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      segs.push({ a, b, len }); total += len;
    }
    const out = [];
    for (let k = 0; k < n; k++) {
      let d = k / n * total, i = 0;
      while (d > segs[i].len) { d -= segs[i].len; i++; }
      const s = segs[i], t = d / s.len;
      out.push([s.a[0] + (s.b[0] - s.a[0]) * t, s.a[1] + (s.b[1] - s.a[1]) * t]);
    }
    return out;
  }
  const EP_N = 512;
  const path = samplePolygon(BOLT, EP_N);
  const coeffs = [];
  for (let k = -EP_N / 2; k < EP_N / 2; k++) {
    let re = 0, im = 0;
    for (let m = 0; m < EP_N; m++) {
      const ang = -2 * Math.PI * k * m / EP_N;
      re += path[m][0] * Math.cos(ang) - path[m][1] * Math.sin(ang);
      im += path[m][0] * Math.sin(ang) + path[m][1] * Math.cos(ang);
    }
    coeffs.push({ k, re: re / EP_N, im: im / EP_N, r: Math.hypot(re, im) / EP_N });
  }
  const zero = coeffs.find(c => c.k === 0);
  const sorted = coeffs.filter(c => c.k !== 0).sort((a, b) => b.r - a.r);
  const EP_K = 40;

  Deck.register('s-fourier', {
    steps: 2,
    init(ctx) {
      this.plot = new Plot(ctx.$('#fs-plot'), {
        w: 920, h: 360, pad: { l: 60, r: 20, t: 16, b: 40 }, font: 22,
        x: { min: 0, max: 1, tickValues: [0, 0.5, 1], grid: false },
        y: { min: -0.85, max: 1.25, tickValues: [0, 1] },
      });
      this.bars = new Plot(ctx.$('#fs-plot').parentNode.appendChild(U.el('canvas', { id: 'fs-bars' })), {
        w: 920, h: 160, pad: { l: 60, r: 20, t: 26, b: 38 }, font: 20,
        x: { min: -0.7, max: KMAX + 0.7, tickValues: [0, 4, 8, 12, 16, 20, 24], grid: false, label: '' },
        y: { min: 0, max: Math.max(...mags) * 1.1, hideTicks: true, grid: false },
      });
      this.epi = ctx.$('#fs-epi');
      this.readout = ctx.$('.fs-readout');
      this.state = { curve: partial(1), K: 1 };
      this.resize();
    },
    resize() {
      this.plot.resize(); this.bars.resize();
      this.ge = U.hidpi(this.epi, 700, 450);
      this.draw();
    },
    draw() {
      const p = this.plot, s = this.state;
      p.clear(); p.drawAxes();
      p.line(xs, ys, { color: '#b9b6ad', width: 7 });
      p.line(xs, s.curve, { color: '#2a78d6', width: 3.5 });
      const b = this.bars, g = b.ctx;
      b.clear();
      g.save();
      g.font = '500 20px "Fira Sans"'; g.fillStyle = '#4b4f5c';
      g.fillText('size of each coefficient pair (aₖ, bₖ)', 60, 14);
      for (let k = 0; k <= KMAX; k++) {
        const x = b.sx(k), y0 = b.sy(0), y1 = b.sy(mags[k]);
        g.fillStyle = k <= s.K ? '#2a78d6' : '#dcd9d0';
        U.roundRect(g, x - 9, y1, 18, Math.max(2, y0 - y1), 4); g.fill();
      }
      g.fillStyle = '#8a8780'; g.textAlign = 'center'; g.font = '400 19px "Fira Sans"';
      for (const k of [0, 4, 8, 12, 16, 20, 24]) g.fillText(k, b.sx(k), b.sy(0) + 22);
      g.textAlign = 'left'; g.fillText('k', b.sx(KMAX) + 20, b.sy(0) + 22);
      g.restore();
      this.readout.textContent = `K = ${s.K} term${s.K === 1 ? '' : 's'}  ·  ${2 * s.K + 1} numbers`;
    },
    runSeries(c) {
      return c.run(async c => {
        await c.wait(400);
        while (true) {
          for (let i = 1; i < KS.length; i++) {
            const from = this.state.curve.slice(), to = partial(KS[i]);
            this.state.K = KS[i];
            await c.play(700, t => { this.state.curve = from.map((v, j) => v + (to[j] - v) * t); this.draw(); });
            await c.wait(i === KS.length - 1 ? 2600 : 650);
          }
          const from = this.state.curve.slice(), to = partial(1);
          this.state.K = 1;
          await c.play(900, t => { this.state.curve = from.map((v, j) => v + (to[j] - v) * t); this.draw(); });
          await c.wait(500);
        }
      });
    },
    runEpicycles(c) {
      if (this.epiOn) return;
      this.epiOn = true;
      const g = this.ge, W = 700, H = 450, cx = W / 2, cy = H / 2, S = 200;
      const terms = sorted.slice(0, EP_K);
      let trace = [];
      const t0 = performance.now() / 1000;
      const PERIOD = 9;
      c.loop(now => {
        const tau = ((now - t0) / PERIOD) % 1;
        if (trace.length && tau < trace[trace.length - 1].tau) trace = [];
        g.clearRect(0, 0, W, H);
        let x = cx + zero.re * S, y = cy + zero.im * S;
        g.lineWidth = 1.2;
        for (const c of terms) {
          const ang = 2 * Math.PI * c.k * tau;
          const nx = x + (c.re * Math.cos(ang) - c.im * Math.sin(ang)) * S;
          const ny = y + (c.re * Math.sin(ang) + c.im * Math.cos(ang)) * S;
          g.strokeStyle = 'rgba(29,33,48,0.16)';
          g.beginPath(); g.arc(x, y, c.r * S, 0, 2 * Math.PI); g.stroke();
          g.strokeStyle = 'rgba(29,33,48,0.55)';
          g.beginPath(); g.moveTo(x, y); g.lineTo(nx, ny); g.stroke();
          x = nx; y = ny;
        }
        trace.push({ x, y, tau });
        // target outline (faint)
        g.strokeStyle = 'rgba(185,182,173,0.55)'; g.lineWidth = 6; g.lineJoin = 'round';
        g.beginPath(); BOLT.forEach(([px, py], i) => (i ? g.lineTo(cx + px * S, cy + py * S) : g.moveTo(cx + px * S, cy + py * S))); g.closePath(); g.stroke();
        // trace
        g.strokeStyle = '#2a78d6'; g.lineWidth = 4; g.lineCap = 'round';
        g.beginPath(); trace.forEach((p, i) => (i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y))); g.stroke();
        g.fillStyle = '#2a78d6'; g.beginPath(); g.arc(x, y, 6, 0, 2 * Math.PI); g.fill();
        // label the pen with z(t), as in the formula (bold z, italic t)
        const parts = [['z', 'bold 26px KaTeX_Main, serif'], ['(', '26px KaTeX_Main, serif'], ['t', 'italic 26px KaTeX_Math, serif'], [')', '26px KaTeX_Main, serif']];
        g.textAlign = 'left'; g.textBaseline = 'middle'; g.lineWidth = 5; g.strokeStyle = '#fbfaf6'; g.lineJoin = 'round';
        let lx = Math.min(x + 14, W - 60);
        for (const [txt, font] of parts) {
          g.font = font;
          g.strokeText(txt, lx, y - 16); g.fillStyle = '#2a78d6'; g.fillText(txt, lx, y - 16);
          lx += g.measureText(txt).width;
        }
      });
    },
    enter(ctx, step) {
      this.epiOn = false;
      this.state = { curve: partial(1), K: 1 };
      this.draw();
      this.runSeries(ctx);
      if (step >= 1) this.runEpicycles(ctx);
      else this.ge.clearRect(0, 0, 700, 450);
    },
    step(ctx, step, prev) {
      if (step >= 1 && prev < 1) this.runEpicycles(ctx);
    },
  });
})();
