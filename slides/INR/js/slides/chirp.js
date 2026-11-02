/* The chirp, live in TensorFlow.js: plain ReLU MLP (slide 6) vs. Fourier features (slide 9) */
(function () {
  const NPTS = 1024;
  const ts = Float32Array.from({ length: NPTS }, (_, i) => i / (NPTS - 1));
  const ys = ts.map(t => Math.sin(2 * Math.PI * (t + 20 * t * t)));
  const GRAY = '#b9b6ad', ORANGE = '#eb6834', BLUE = '#2a78d6';
  const FMAX = 60;

  /* magnitude spectrum with a light Tukey taper */
  function spectrum(sig) {
    const n = 1024, re = new Float64Array(n), im = new Float64Array(n);
    const a = 0.1;
    for (let i = 0; i < n; i++) {
      const x = i / (n - 1);
      let w = 1;
      if (x < a / 2) w = 0.5 * (1 + Math.cos(2 * Math.PI / a * (x - a / 2)));
      else if (x > 1 - a / 2) w = 0.5 * (1 + Math.cos(2 * Math.PI / a * (x - 1 + a / 2)));
      re[i] = sig[i] * w;
    }
    U.fft(re, im);
    const out = new Float64Array(FMAX + 1);
    for (let k = 0; k <= FMAX; k++) out[k] = Math.hypot(re[k], im[k]) / (n / 2);
    return out;
  }
  const targetSpec = spectrum(ys);
  const freqs = Array.from({ length: FMAX + 1 }, (_, k) => k);
  const specMax = Math.max(...targetSpec) * 1.25;

  const makeRelu = () => new INR.SignalFit(new INR.Model({ inDim: 1, outDim: 1, width: 256, depth: 3, outAct: 'linear', seed: 11 }), ts, ys, 1e-3);
  const makeBig = () => new INR.SignalFit(new INR.Model({ inDim: 1, outDim: 1, width: 512, depth: 4, outAct: 'linear', seed: 13 }), ts, ys, 5e-4);
  const makeFF = () => new INR.SignalFit(new INR.Model({ inDim: 1, outDim: 1, features: 'fourier', m: 64, sigma: 20, width: 64, depth: 3, outAct: 'linear', seed: 5 }), ts, ys, 2.5e-4);
  INR.addWarmup(() => INR.warmTrainer(makeRelu()));
  INR.addWarmup(() => INR.warmTrainer(makeBig()));
  INR.addWarmup(() => INR.warmTrainer(makeFF()));

  /* slide 6: two plain ReLU MLPs as fast as possible; slide 9: ReLU vs Fourier features, paced so
     the audience can follow (one step per model every ~110 ms)                                    */
  const CONFIGS = {
    's-chirp': { models: [['relu', makeRelu, '#eb6834'], ['big', makeBig, '#8a3008']], maxSteps: 3000, paced: false, ticks: [0, 1000, 2000, 3000] },
    's-chirp-ff': { models: [['relu', makeRelu, '#eb6834'], ['ff', makeFF, BLUE]], maxSteps: 800, paced: true, ticks: [0, 200, 400, 600, 800] },
  };

  function makeSlide(id) {
    const cfg = CONFIGS[id];
    Deck.register(id, {
      steps: id === 's-chirp' ? 1 : 0,
      init(ctx) {
        this.main = new Plot(ctx.$('.chirp-main'), {
          w: 1700, h: 330, pad: { l: 60, r: 20, t: 14, b: 44 }, font: 22,
          x: { min: 0, max: 1, tickValues: [0, 0.25, 0.5, 0.75, 1], grid: false },
          y: { min: -1.3, max: 1.3, tickValues: [-1, 0, 1] },
        });
        this.loss = new Plot(ctx.$('.chirp-loss'), {
          w: 560, h: 262, pad: { l: 74, r: 24, t: 10, b: 52 }, font: 20,
          x: { min: 0, max: cfg.maxSteps, tickValues: cfg.ticks, label: 'step', grid: false },
          y: { min: 1e-1, max: 1, log: true },
        });
        this.spec = new Plot(ctx.$('.chirp-spec'), {
          w: 560, h: 262, pad: { l: 30, r: 24, t: 10, b: 52 }, font: 20,
          x: { min: 0, max: FMAX, tickValues: [0, 20, 40, 60], label: 'frequency (cycles per unit)', grid: false },
          y: { min: 0, max: specMax, tickValues: [0], hideTicks: true, grid: false },
        });
        this.models = [];
        this.preds = {};
        this.draw();
      },
      resize() { this.main.resize(); this.loss.resize(); this.spec.resize(); this.draw(); },
      /* the loss axis follows the run: it extends downwards (smoothly) as the loss drops */
      updateAxis() {
        let mn = Infinity;
        for (const tr of this.models) for (let i = Math.max(0, tr.hist.loss.length - 100); i < tr.hist.loss.length; i++) mn = Math.min(mn, tr.hist.loss[i]);
        if (!isFinite(mn)) return;
        const want = Math.min(-1, Math.floor(Math.log10(mn * 0.8)));
        const cur = Math.log10(this.loss.o.y.min);
        const nl = cur + (want - cur) * 0.15;
        this.loss.o.y.min = Math.pow(10, Math.abs(nl - want) < 0.01 ? want : nl);
        this.loss.o.y.tickEvery = (-Math.log10(this.loss.o.y.min)) > 4 ? 2 : 1;
      },
      draw() {
        const m = this.main, L = this.loss, S = this.spec;
        m.clear(); m.drawAxes();
        m.line(ts, ys, { color: GRAY, width: 6 });
        for (const tr of this.models) if (this.preds[tr.key]) m.line(ts, this.preds[tr.key], { color: tr.color, width: 3.2 });
        // frequency annotation
        m.text('low frequency', m.sx(0.02), 30, { color: '#8a8780', font: '500 21px "Fira Sans"' });
        m.text('high frequency', m.sx(0.98), 30, { color: '#8a8780', font: '500 21px "Fira Sans"', align: 'right' });

        this.updateAxis();
        L.clear(); L.drawAxes();
        for (const tr of this.models) {
          const h = tr.hist;
          L.line(h.steps, h.loss, { color: tr.color, width: 3 });
          if (h.steps.length) L.dot(h.steps[h.steps.length - 1], h.loss[h.loss.length - 1], { r: 5, color: tr.color });
        }
        S.clear(); S.drawAxes();
        S.area(freqs, Array.from(targetSpec), 0, { color: 'rgba(185,182,173,0.45)' });
        S.line(freqs, Array.from(targetSpec), { color: GRAY, width: 2.5 });
        for (const tr of this.models) if (this.preds[tr.key]) S.line(freqs, Array.from(spectrum(this.preds[tr.key])), { color: tr.color, width: 3 });
      },
      enter(ctx) {
        this.preds = {};
        this.loss.o.y.min = 1e-1;
        this.draw();
        ctx.run(async c => {
          await INR.init();
          await c.sleep(500);
          this.dispose();
          this.models = cfg.models.map(([key, make, color]) => Object.assign(make(), { key, color }));
          this.models.forEach(m => { m.stepsPerChunk = cfg.paced ? 1 : 8; });
          this.stop = INR.run(c, this.models, {
            targetMs: 30, maxSteps: cfg.paced ? 1 : 40, minRoundMs: cfg.paced ? 110 : 0,
            stopWhen: tr => tr.step >= cfg.maxSteps,
            onFrame: async trs => {
              for (const tr of trs) this.preds[tr.key] = await tr.predict();
              this.draw();
            },
          });
          // keep easing the axis after the runs end
          c.loop(() => { if (this.models.length && this.models.every(t => t.step >= cfg.maxSteps)) this.draw(); });
        });
      },
      leave() { if (this.stop) this.stop(); this.dispose(); },
      dispose() {
        const ms = this.models;
        this.models = [];
        ms.forEach(m => m.dispose());          // deferred internally while a chunk is in flight
      },
    });
  }
  makeSlide('s-chirp');
  makeSlide('s-chirp-ff');
})();
