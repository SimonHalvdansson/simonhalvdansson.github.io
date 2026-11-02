/* Computed tomography: measurement animation, INR-for-CT concept, and live sparse-view reconstruction */
(function () {
  /* ---------------- modified Shepp–Logan phantom: (rho, a, b, x0, y0, phi_deg) ---------------- */
  const E = [[1.0, .69, .92, 0, 0, 0], [-.8, .6624, .8740, 0, -.0184, 0], [-.2, .1100, .3100, .22, 0, -18],
    [-.2, .1600, .4100, -.22, 0, 18], [.1, .2100, .2500, 0, .35, 0], [.1, .0460, .0460, 0, .1, 0],
    [.1, .0460, .0460, 0, -.1, 0], [.1, .0460, .0230, -.08, -.605, 0], [.1, .0230, .0230, 0, -.606, 0],
    [.1, .0230, .0460, .06, -.605, 0]];
  const WIN = [0, 0.6];   // display window

  function phantom(R) {
    const img = new Float32Array(R * R);
    for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
      const x = -1 + 2 * j / (R - 1), y = 1 - 2 * i / (R - 1);
      let v = 0;
      for (const [rho, a, b, x0, y0, ph] of E) {
        const p = ph * Math.PI / 180, c = Math.cos(p), s = Math.sin(p);
        const xr = (x - x0) * c + (y - y0) * s, yr = -(x - x0) * s + (y - y0) * c;
        if ((xr / a) ** 2 + (yr / b) ** 2 <= 1) v += rho;
      }
      img[i * R + j] = v;
    }
    return img;
  }
  /* analytic line integrals: p(theta, s) = int mu(s n + t n_perp) dt */
  function radon(thetas, ss) {
    const P = new Float32Array(thetas.length * ss.length);
    thetas.forEach((th, i) => {
      for (const [rho, a, b, x0, y0, ph] of E) {
        const al = ph * Math.PI / 180;
        const r2 = (a * Math.cos(th - al)) ** 2 + (b * Math.sin(th - al)) ** 2;
        const off = x0 * Math.cos(th) + y0 * Math.sin(th);
        ss.forEach((s, j) => {
          const sp = s - off;
          if (sp * sp <= r2) P[i * ss.length + j] += 2 * rho * a * b / r2 * Math.sqrt(r2 - sp * sp);
        });
      }
    });
    return P;
  }
  /* filtered back-projection with the Ram-Lak kernel (Kak & Slaney) */
  function fbp(P, thetas, ss, R) {
    const n = ss.length, ds = ss[1] - ss[0], na = thetas.length;
    const h = new Float32Array(2 * n - 1);
    for (let k = -(n - 1); k <= n - 1; k++) {
      h[k + n - 1] = k === 0 ? 1 / (4 * ds * ds) : (k % 2 ? -1 / (Math.PI * Math.PI * k * k * ds * ds) : 0);
    }
    const Q = new Float32Array(na * n);
    for (let a = 0; a < na; a++) for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let m = 0; m < n; m++) acc += P[a * n + m] * h[j - m + n - 1];
      Q[a * n + j] = acc * ds;
    }
    const img = new Float32Array(R * R);
    for (let a = 0; a < na; a++) {
      const c = Math.cos(thetas[a]), s = Math.sin(thetas[a]);
      for (let i = 0; i < R; i++) {
        const y = 1 - 2 * i / (R - 1);
        for (let j = 0; j < R; j++) {
          const x = -1 + 2 * j / (R - 1);
          const t = (x * c + y * s - ss[0]) / ds;
          const k = Math.floor(t);
          if (k >= 0 && k < n - 1) { const f = t - k; img[i * R + j] += Q[a * n + k] * (1 - f) + Q[a * n + k + 1] * f; }
        }
      }
    }
    for (let i = 0; i < img.length; i++) img[i] *= Math.PI / na;
    return img;
  }
  const psnr = (a, b) => {
    let e = 0;
    for (let i = 0; i < a.length; i++) { const d = Math.min(1, Math.max(0, a[i])) - b[i]; e += d * d; }
    return U.psnr(e / a.length);
  };
  const linspace = (a, b, n, endpoint = true) => Float32Array.from({ length: n }, (_, i) => a + (b - a) * i / (endpoint ? n - 1 : n));
  /* gray image with a window into a canvas (offscreen, R x R) */
  function grayCanvas(img, w, h, lo = WIN[0], hi = WIN[1]) {
    const c = document.createElement('canvas');
    U.putGray(c, img, w, h, lo, hi);
    return c;
  }
  /* sinogram (na x ns, row-major by angle) to a canvas with angle on x, detector on y */
  function sinoCanvas(P, na, ns, max) {
    const T = new Float32Array(na * ns);
    for (let a = 0; a < na; a++) for (let j = 0; j < ns; j++) T[(ns - 1 - j) * na + a] = P[a * ns + j];
    return grayCanvas(T, na, ns, 0, max);
  }
  const CT = { phantom, radon, fbp, psnr, linspace, grayCanvas, sinoCanvas, WIN };

  /* =============================== slide 12: measurement =============================== */
  Deck.register('s-ct', {
    steps: 2,
    init(ctx) {
      this.cs = ctx.$('#ct-scene'); this.cg = ctx.$('#ct-sino');
      this.R = 220;
      this.ph = grayCanvas(phantom(this.R), this.R, this.R);
      this.NA = 180; this.NS = 160;
      this.thetas = linspace(0, Math.PI, this.NA, false);
      this.ss = linspace(-1, 1, this.NS);
      this.P = radon(this.thetas, this.ss);
      this.pmax = Math.max(...this.P);
      this.sino = document.createElement('canvas'); this.sino.width = this.NA; this.sino.height = this.NS;
      this.resize();
    },
    resize() {
      this.gs = U.hidpi(this.cs, 740, 740);
      this.gg = U.hidpi(this.cg, 620, 340);
    },
    sinoUpTo(k) {
      const g = this.sino.getContext('2d'), im = g.createImageData(this.NA, this.NS);
      for (let a = 0; a < this.NA; a++) for (let j = 0; j < this.NS; j++) {
        const o = ((this.NS - 1 - j) * this.NA + a) * 4;
        const v = a <= k ? Math.min(255, this.P[a * this.NS + j] / this.pmax * 255) : 0;
        im.data[o] = im.data[o + 1] = im.data[o + 2] = v; im.data[o + 3] = a <= k ? 255 : 0;
      }
      g.putImageData(im, 0, 0);
    },
    drawScene(phi, k) {
      const g = this.gs, W = 740, C = W / 2, S = 195;      // S: pixels per unit
      g.clearRect(0, 0, W, W);
      // phantom, clipped to the field of view
      g.save();
      g.beginPath(); g.arc(C, C, S * 1.0, 0, 2 * Math.PI); g.fillStyle = '#000'; g.fill(); g.clip();
      g.imageSmoothingEnabled = true;
      g.drawImage(this.ph, C - S, C - S, 2 * S, 2 * S);
      g.restore();
      const n = [Math.cos(phi), Math.sin(phi)], np = [-Math.sin(phi), Math.cos(phi)];
      // screen coords: y up -> flip
      const P = (x, y) => [C + x * S, C - y * S];
      const T = 1.32;
      // rays
      g.lineWidth = 1.6;
      for (let r = 0; r <= 24; r++) {
        const s = -1 + 2 * r / 24;
        const [x0, y0] = P(s * n[0] - T * np[0], s * n[1] - T * np[1]);
        const [x1, y1] = P(s * n[0] + T * np[0], s * n[1] + T * np[1]);
        const grad = g.createLinearGradient(x0, y0, x1, y1);
        grad.addColorStop(0, 'rgba(237,161,0,0.85)'); grad.addColorStop(1, 'rgba(237,161,0,0.25)');
        g.strokeStyle = grad;
        g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      }
      // source (behind) and detector (front)
      const src0 = P(-1.1 * n[0] - (T + 0.06) * np[0], -1.1 * n[1] - (T + 0.06) * np[1]);
      const src1 = P(1.1 * n[0] - (T + 0.06) * np[0], 1.1 * n[1] - (T + 0.06) * np[1]);
      g.strokeStyle = '#eda100'; g.lineWidth = 12; g.lineCap = 'round';
      g.beginPath(); g.moveTo(...src0); g.lineTo(...src1); g.stroke();
      const d0 = P(-1.1 * n[0] + (T + 0.04) * np[0], -1.1 * n[1] + (T + 0.04) * np[1]);
      const d1 = P(1.1 * n[0] + (T + 0.04) * np[0], 1.1 * n[1] + (T + 0.04) * np[1]);
      g.strokeStyle = '#1d2130'; g.lineWidth = 14;
      g.beginPath(); g.moveTo(...d0); g.lineTo(...d1); g.stroke();
      // projection profile on the detector
      const kk = Math.min(this.NA - 1, k);
      g.strokeStyle = '#2a78d6'; g.lineWidth = 4; g.lineJoin = 'round'; g.lineCap = 'round';
      g.beginPath();
      for (let j = 0; j < this.NS; j++) {
        const s = this.ss[j], v = this.P[kk * this.NS + j] / this.pmax * 0.28;
        const [x, y] = P(s * n[0] + (T + 0.1 + v) * np[0], s * n[1] + (T + 0.1 + v) * np[1]);
        j ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
      g.font = 'italic 28px KaTeX_Math, serif'; g.fillStyle = '#fff';
      g.fillText('μ(x, y)', C, C + S * 0.8);
    },
    drawSino(k) {
      const g = this.gg, W = 620, H = 340;
      g.clearRect(0, 0, W, H);
      g.fillStyle = '#ebe8e0'; U.roundRect(g, 0, 0, W, H, 12); g.fill();
      this.sinoUpTo(k);
      g.save(); U.roundRect(g, 0, 0, W, H, 12); g.clip();
      g.imageSmoothingEnabled = true;
      g.drawImage(this.sino, 0, 0, W, H);
      g.restore();
      if (k < this.NA) { const x = (k + 0.5) / this.NA * W; g.fillStyle = '#2a78d6'; g.fillRect(x - 1.5, 0, 3, H); }
    },
    enter(ctx) {
      ctx.run(async c => {
        while (true) {
          await c.play(9000, (e, raw) => {
            const phi = raw * Math.PI, k = Math.floor(raw * this.NA);
            this.drawScene(phi, k); this.drawSino(k);
          }, U.Ease.linear);
          this.drawSino(this.NA);
          await c.wait(2500);
        }
      });
    },
  });

  /* =============================== slide 13: concept diagram =============================== */
  Deck.register('s-ct-inr', {
    steps: 3,
    init(ctx) {
      const svg = this.svg = ctx.$('#cti-svg');
      const R = 160;
      const gt = phantom(R);
      const NA = 30, NS = 120;
      const th = linspace(0, Math.PI, NA, false), ss = linspace(-1, 1, NS);
      const Py = radon(th, ss);
      const pmax = Math.max(...Py);
      // a deliberately imperfect "current guess": smoothed phantom
      const guess = new Float32Array(R * R);
      const k = 5;
      for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
        let acc = 0, cnt = 0;
        for (let di = -k; di <= k; di += 1) for (let dj = -k; dj <= k; dj += 1) {
          const ii = i + di, jj = j + dj;
          if (ii >= 0 && ii < R && jj >= 0 && jj < R) { acc += gt[ii * R + jj]; cnt++; }
        }
        guess[i * R + j] = 0.85 * acc / cnt;
      }
      // projections of the guess, numerically (rotate-and-sum via sampling)
      const Pg = new Float32Array(NA * NS);
      for (let a = 0; a < NA; a++) {
        const c = Math.cos(th[a]), s = Math.sin(th[a]);
        for (let j = 0; j < NS; j++) {
          let acc = 0;
          for (let q = 0; q < 120; q++) {
            const t = -1 + 2 * (q + 0.5) / 120;
            const x = ss[j] * c - t * s, y = ss[j] * s + t * c;
            const jj = Math.round((x + 1) / 2 * (R - 1)), ii = Math.round((1 - y) / 2 * (R - 1));
            if (ii >= 0 && ii < R && jj >= 0 && jj < R) acc += guess[ii * R + jj];
          }
          Pg[a * NS + j] = acc * 2 / 120;
        }
      }
      const url = c => c.toDataURL();
      const imGuess = url(grayCanvas(guess, R, R));
      const imY = url(sinoCanvas(Py, NA, NS, pmax));
      const imYh = url(sinoCanvas(Pg, NA, NS, pmax));

      const defs0 = U.svg('defs', {}, svg);
      const g = U.svg('g', {}, svg);
      const box = (x, y, w, h, fill = '#ffffff', stroke = 'rgba(29,33,48,0.18)') => U.svg('rect', { x, y, width: w, height: h, rx: 18, fill, stroke, 'stroke-width': 1.5 }, g);
      /* text with simple subscripts: "f_θ" -> f with a subscript θ */
      const text = (x, y, s, opts = {}) => {
        const t = U.svg('text', { x, y, 'text-anchor': opts.anchor || 'middle', 'font-size': opts.size || 26, fill: opts.fill || '#1d2130',
          'font-family': opts.font || 'Fira Sans', 'font-weight': opts.weight || 400, 'font-style': opts.italic ? 'italic' : 'normal' }, g);
        s.split(/(_[^\s(),‖²]+)/).forEach(part => {
          if (!part) return;
          const sub = part[0] === '_';
          const ts = U.svg('tspan', sub ? { 'baseline-shift': 'sub', 'font-size': '70%' } : {}, t);
          ts.textContent = sub ? part.slice(1) : part;
        });
        return t;
      };
      let clipN = 0;
      const img = (href, x, y, w, h, pix = false) => {
        // rounded corners matching the outline drawn on top
        const id = 'cti-clip-' + (clipN++);
        const cp = U.svg('clipPath', { id }, defs0);
        U.svg('rect', { x, y, width: w, height: h, rx: 12 }, cp);
        return U.svg('image', { href, x, y, width: w, height: h, preserveAspectRatio: 'none', 'clip-path': `url(#${id})`, style: pix ? 'image-rendering:pixelated' : '' }, g);
      };
      // layout
      const Y = 70, H = 220;
      // theta
      box(0, Y + 50, 150, 120, '#eef4fc', 'rgba(42,120,214,0.35)');
      text(75, Y + 105, 'θ', { size: 46, font: 'KaTeX_Math, serif', italic: true, fill: '#2a78d6' });
      text(75, Y + 145, 'weights', { size: 22, fill: '#4b4f5c' });
      // network box
      box(220, Y, 260, H);
      const mini = U.svg('svg', { x: 240, y: Y + 30, width: 220, height: 130, viewBox: '0 0 220 130', overflow: 'visible' }, g);
      this.net = new MLPDiagram(mini, { layers: [2, 5, 5, 1], x: 14, y: 6, w: 190, h: 118, r: 8.5, edgeWidth: 1.1 });
      text(350, Y + 196, 'f_θ(x, y)', { size: 28, font: 'KaTeX_Main, serif', italic: true });
      // guess image
      img(imGuess, 560, Y, H, H);
      U.svg('rect', { x: 560, y: Y, width: H, height: H, rx: 12, fill: 'none', stroke: '#2a78d6', 'stroke-width': 3 }, g);
      // forward operator
      box(860, Y + 40, 200, 140, '#fff8e8', 'rgba(237,161,0,0.6)');
      text(960, Y + 110, 'A', { size: 46, font: 'KaTeX_Math, serif', italic: true, fill: '#b07800' });
      text(960, Y + 150, 'integrate along rays', { size: 20, fill: '#7a5a00' });
      // predicted sinogram
      img(imYh, 1140, Y, 170, H);
      U.svg('rect', { x: 1140, y: Y, width: 170, height: H, rx: 12, fill: 'none', stroke: '#2a78d6', 'stroke-width': 3 }, g);
      // measured sinogram
      img(imY, 1500, Y, 170, H);
      U.svg('rect', { x: 1500, y: Y, width: 170, height: H, rx: 12, fill: 'none', stroke: '#1d2130', 'stroke-width': 3 }, g);
      // compare
      U.svg('circle', { cx: 1405, cy: Y + H / 2, r: 30, fill: '#fff', stroke: '#d64545', 'stroke-width': 2.5 }, g);
      text(1405, Y + H / 2 + 11, '−', { size: 40, fill: '#d64545' });
      // forward arrows
      const fw = [[152, Y + 110, 216, Y + 110], [482, Y + 110, 554, Y + 110], [782, Y + 110, 854, Y + 110], [1062, Y + 110, 1134, Y + 110], [1312, Y + 110, 1372, Y + 110], [1496, Y + 110, 1438, Y + 110]];
      const defs = U.svg('defs', {}, svg);
      const mk = (id, color) => {
        const m = U.svg('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
        U.svg('path', { d: 'M0,0 L10,5 L0,10 z', fill: color }, m);
      };
      mk('arr-ink', '#4b4f5c'); mk('arr-red', '#d64545');
      this.fwd = fw.map(([x0, y0, x1, y1]) => U.svg('line', { x1: x0, y1: y0, x2: x1, y2: y1, stroke: '#4b4f5c', 'stroke-width': 3, 'marker-end': 'url(#arr-ink)' }, g));
      // backward (gradient) path
      const back = U.svg('path', {
        d: `M1405 ${Y + H / 2 + 34} C 1405 ${Y + 330}, 1100 ${Y + 336}, 760 ${Y + 336} S 90 ${Y + 326}, 75 ${Y + 176}`,
        fill: 'none', stroke: '#d64545', 'stroke-width': 3.5, 'marker-end': 'url(#arr-red)',
      }, g);
      this.back = back;
      this.backLen = back.getTotalLength();
      back.style.strokeDasharray = this.backLen;
      back.style.strokeDashoffset = this.backLen;
      // math captions as HTML (KaTeX) on top of the SVG; the SVG is drawn at 1 unit = 1 css px
      const wrap = ctx.$('.cti-pipe');
      wrap.style.position = 'relative';
      const lbl = (x, y, html, color, cls = '') => {
        const d = U.el('div', { class: 'cti-lbl2 ' + cls, style: `left:${x}px;top:${y + 18}px;color:${color}` }, wrap, html);
        U.typeset(d);
        return d;
      };
      lbl(670, Y + H + 14, 'current guess \\( \\mu_\\theta = f_\\theta \\)', '#2a78d6');
      lbl(1225, Y + H + 14, 'predicted \\( \\hat y = A(f_\\theta) \\)', '#2a78d6');
      lbl(1585, Y + H + 14, 'measured \\( y \\)', '#1d2130');
      lbl(1405, Y - 52, '\\( \\| \\hat y - y \\|^2 \\)', '#d64545');
      this.backLbl = lbl(760, Y + 344, 'gradient \\( \\nabla_\\theta \\| A(f_\\theta) - y \\|^2 \\) flows back through \\( A \\)', '#d64545', 'strong');
      this.backLbl.style.opacity = 0;
    },
    enter(ctx, step) { this.setBack(step >= 2 ? 1 : 0); if (step >= 2) this.loopPulse(ctx); },
    setBack(t) {
      this.back.style.strokeDashoffset = this.backLen * (1 - t);
      this.backLbl.style.opacity = t;
    },
    loopPulse(ctx) {
      if (this.pulsing) return;
      this.pulsing = true;
      ctx.run(async c => {
        try {
          while (true) { await this.net.pulse(c, { dur: 420, color: '#2a78d6' }); await c.wait(1600); }
        } finally { this.pulsing = false; }
      });
    },
    step(ctx, step, prev) {
      if (step >= 2 && prev < 2) {
        ctx.run(async c => {
          await c.wait(350);
          await c.play(1400, t => this.setBack(t));
        });
        this.loopPulse(ctx);
      } else if (step < 2) this.setBack(0);
    },
    leave() { this.pulsing = false; },
  });

  /* =============================== slide 14: live sparse-view CT =============================== */
  const R = 160, NS = 192, MAX_STEPS = 1200;
  const makeModel = () => new INR.Model({ inDim: 2, outDim: 1, features: 'fourier', m: 128, sigma: 4, width: 128, depth: 3, outAct: 'sigmoid', outBias: -2, seed: 9 });
  const makeTrainer = (thetas, ss, sino) => new INR.CTFit(makeModel(), { thetas, ss, sino, nray: 512, nt: 64, lr: 3e-3, lrEnd: 1e-4, steps: MAX_STEPS, seed: 5 });
  INR.addWarmup(async () => {
    const th = linspace(0, Math.PI, 30, false), ss = linspace(-1, 1, NS);
    await INR.warmTrainer(makeTrainer(th, ss, new Float32Array(30 * NS)), 'render', R);
  });

  Deck.register('s-ct-live', {
    init(ctx) {
      this.c = { gt: ctx.$('#ctl-gt'), sino: ctx.$('#ctl-sino'), fbp: ctx.$('#ctl-fbp'), inr: ctx.$('#ctl-inr') };
      this.nEl = ctx.$('.ctl-n'); this.stepEl = ctx.$('.ctl-step span');
      this.pf = ctx.$('.ctl-psnr-fbp'); this.pi = ctx.$('.ctl-psnr-inr');
      this.gt = phantom(R);
      this.gtC = grayCanvas(this.gt, R, R);
      this.plot = new Plot(ctx.$('#ctl-plot'), {
        w: 1260, h: 250, pad: { l: 96, r: 30, t: 12, b: 56 }, font: 21,
        x: { min: 0, max: MAX_STEPS, tickValues: [0, 300, 600, 900, 1200], label: 'training step', grid: false },
        y: { min: 1e-5, max: 1e-1, log: true, label: 'loss on projections', labelX: 4 },
      });
      this.nang = 30;
      this.resize();
    },
    resize() {
      this.g = {};
      for (const k in this.c) this.g[k] = U.hidpi(this.c[k], 380, 380);
      this.plot.resize();
      this.drawAll();
    },
    draw(key, src, pix = false) {
      const g = this.g[key];
      g.save(); U.roundRect(g, 0, 0, 380, 380, 14); g.clip();
      g.fillStyle = '#000'; g.fillRect(0, 0, 380, 380);
      g.imageSmoothingEnabled = !pix;
      g.drawImage(src, 0, 0, 380, 380);
      g.restore();
    },
    setup(nang) {
      this.nang = nang;
      this.nEl.textContent = nang;
      this.thetas = linspace(0, Math.PI, nang, false);
      this.ss = linspace(-1, 1, NS);
      this.sino = radon(this.thetas, this.ss);
      this.smax = Math.max(...this.sino);
      this.sinoC = sinoCanvas(this.sino, nang, NS, this.smax);
      const f = fbp(this.sino, this.thetas, this.ss, R);
      this.fbpC = grayCanvas(f, R, R);
      this.pf.textContent = `${psnr(f, this.gt).toFixed(1)} dB`;
      this.pi.textContent = '';
      this.inrC = null;
      this.hist = { steps: [], loss: [] };
      this.drawAll();
    },
    drawAll() {
      if (!this.g) return;
      this.draw('gt', this.gtC);
      if (this.sinoC) this.draw('sino', this.sinoC, true);
      if (this.fbpC) this.draw('fbp', this.fbpC);
      if (this.inrC) this.draw('inr', this.inrC); else if (this.g.inr) { const g = this.g.inr; g.clearRect(0, 0, 380, 380); g.fillStyle = '#000'; U.roundRect(g, 0, 0, 380, 380, 14); g.fill(); }
      this.drawPlot();
    },
    drawPlot() {
      const p = this.plot, h = this.hist || { steps: [] };
      p.clear(); p.drawAxes();
      if (h.steps.length > 1) {
        // light smoothing for readability (minibatch loss is noisy)
        const sm = []; let e = h.loss[0];
        for (const v of h.loss) { e = 0.9 * e + 0.1 * v; sm.push(e); }
        p.line(h.steps, h.loss, { color: '#2a78d6', width: 1.5, alpha: 0.25 });
        p.line(h.steps, sm, { color: '#2a78d6', width: 3.2 });
      }
    },
    start(ctx) {
      this.stopTraining();
      this.setup(this.nang);
      ctx.run(async c => {
        let backend = null;
        try { backend = await INR.init(); } catch (e) { backend = null; }
        if (!backend) return;
        await c.sleep(500);
        const tr = this.tr = makeTrainer(this.thetas, this.ss, this.sino);
        tr.stepsPerChunk = 4;
        this.hist = tr.hist;
        let frame = 0;
        this.stop = INR.run(c, [tr], {
          targetMs: 50, maxSteps: 32,
          stopWhen: t => t.step >= MAX_STEPS,
          onFrame: async () => {
            if (frame++ % 2 === 0 || tr.step >= MAX_STEPS) {
              const img = await tr.render(R);
              this.inrC = grayCanvas(img, R, R);
              this.draw('inr', this.inrC);
              this.pi.textContent = `${psnr(img, this.gt).toFixed(1)} dB`;
            }
            this.drawPlot();
            this.stepEl.textContent = U.fmtInt(tr.step);
          },
        });
      });
    },
    stopTraining() {
      if (this.stop) { this.stop(); this.stop = null; }
      if (this.tr) { this.tr.dispose(); this.tr = null; }
    },
    enter(ctx) { this.start(ctx); },
    keys: ['1', '2', '3'],
    key(ctx, k) {
      const m = { '1': 15, '2': 30, '3': 90 }[k];
      if (m) { ctx.cancel(); this.nang = m; this.start(ctx); }
    },
    leave() { this.stopTraining(); },
  });

  window.CT = CT;
})();
