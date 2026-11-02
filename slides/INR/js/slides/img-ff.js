/* Attempt 2: Fourier features + MLP learning the same image, live (with a precomputed fallback) */
(function () {
  const RAMP = ['#f2a072', '#eb6834', '#c04a17', '#8a3008'];
  const TILE = 330, R = 128, MAX_STEPS = 2000;
  const makeTrainer = target => {
    const model = new INR.Model({ inDim: 2, outDim: 3, features: 'fourier', m: 128, sigma: 10, width: 128, depth: 2, seed: 3 });
    const tr = new INR.ImageFit(model, target, R, 3e-3);
    tr.lrEnd = 3e-4; tr.decaySteps = MAX_STEPS;
    return tr;
  };
  INR.addWarmup(() => INR.warmTrainer(makeTrainer(new Float32Array(R * R * 3))));

  Deck.register('s-img-ff', {
    init(ctx) {
      const A = this.A = ASSETS.relu;
      this.cT = ctx.$('#iff-target'); this.cR = ctx.$('#iff-relu'); this.cF = ctx.$('#iff-ff');
      this.stepsEl = ctx.$('.iff-steps'); this.timeEl = ctx.$('.iff-time');
      this.psnrEl = ctx.$('.iff-psnr');
      this.off = document.createElement('canvas'); this.off.width = R; this.off.height = R;
      this.plot = new Plot(ctx.$('#iff-plot'), {
        w: 1700, h: 330, pad: { l: 96, r: 150, t: 16, b: 62 }, font: 22,
        x: { min: 0, max: A.steps, tickValues: [0, 1000, 2000, 3000, 4000, 5000], label: 'training step', grid: false },
        y: { min: 1e-3, max: 1e-1, log: true, label: 'Mean squared error (MSE)', labelX: 6 },
      });
      const last = A.runs[A.runs.length - 1].frames;
      this.ready = Promise.all([
        U.loadImage(A.target).then(im => { this.target = im; this.targetRGB = U.imageToRGB(im, R); }),
        U.loadImage(last[last.length - 1].src).then(im => { this.reluImg = im; }),
        Promise.all(A.ff.frames.map(f => U.loadImage(f.src))).then(ims => { this.ffFrames = ims; }),
      ]);
      this.hist = { steps: [], loss: [] };
      this.resize();
    },
    resize() {
      this.gT = U.hidpi(this.cT, TILE, TILE);
      this.gR = U.hidpi(this.cR, TILE, TILE);
      this.gF = U.hidpi(this.cF, TILE, TILE);
      this.plot.resize();
      this.drawStatic(); this.drawPlot();
      if (this.lastPred) this.drawPred(this.lastPred);
    },
    drawImg(g, im) {
      g.save(); U.roundRect(g, 0, 0, TILE, TILE, 14); g.clip();
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(im, 0, 0, TILE, TILE); g.restore();
    },
    drawStatic() {
      if (this.target) this.drawImg(this.gT, this.target);
      if (this.reluImg) this.drawImg(this.gR, this.reluImg);
    },
    drawPred(rgb) {
      this.lastPred = rgb;
      U.putRGB(this.off, rgb, R);
      this.drawImg(this.gF, this.off);
    },
    drawPlot() {
      const p = this.plot, A = this.A, h = this.hist;
      p.clear(); p.drawAxes();
      A.runs.forEach((r, i) => p.line(A.lossSteps, r.loss, { color: RAMP[i], width: 2.2, alpha: 0.45 }));
      const lastRelu = A.runs[A.runs.length - 1];
      p.endLabel('ReLU 791k', A.steps, lastRelu.loss[lastRelu.loss.length - 1], RAMP[3], { dx: 16 });
      if (h.steps.length > 1) {
        const n = h.steps.length;
        p.line(h.steps, h.loss, { color: '#2a78d6', width: 3.2 });
        p.dot(h.steps[n - 1], h.loss[n - 1], { r: 6, color: '#2a78d6' });
        p.endLabel('Fourier features 50k', h.steps[n - 1], h.loss[n - 1], '#2a78d6', { dx: 16 });
      }
    },
    /* smoothly extend the y-axis downwards when the loss goes below it */
    updateAxis(dt) {
      const h = this.hist;
      if (!h.loss.length) return;
      let mn = Infinity;
      for (let i = Math.max(0, h.loss.length - 200); i < h.loss.length; i++) mn = Math.min(mn, h.loss[i]);
      const want = Math.min(1e-3, Math.pow(10, Math.floor(Math.log10(mn * 0.7))));
      const cur = this.plot.o.y.min;
      if (Math.abs(Math.log10(cur) - Math.log10(want)) > 0.01) {
        const nl = U.lerp(Math.log10(cur), Math.log10(want), Math.min(1, dt * 3));
        this.plot.setY(Math.pow(10, nl), 1e-1);
      }
    },
    reset() {
      this.hist = { steps: [], loss: [] };
      this.plot.setY(1e-3, 1e-1);
      this.lastPred = null;
      this.gF.clearRect(0, 0, TILE, TILE);
      this.stepsEl.textContent = '0'; this.timeEl.textContent = '0.0'; this.psnrEl.textContent = '—';
    },
    enter(ctx, step) {
      this.reset();
      ctx.run(async c => {
        await this.ready;
        this.drawStatic(); this.drawPlot();
        let backend = null;
        try { backend = await INR.init(); } catch (e) { backend = null; }
        if (!backend || backend === 'cpu') return this.replay(c);
        await c.sleep(600);
        this.dispose();
        const tr = this.tr = makeTrainer(this.targetRGB);
        tr.stepsPerChunk = 6;
        this.hist = tr.hist;
        const t0 = performance.now();
        let last = t0;
        this.stop = INR.run(c, [tr], {
          targetMs: 55, maxSteps: 64,
          stopWhen: t => t.step >= MAX_STEPS,
          onFrame: async () => {
            const now = performance.now();
            this.drawPred(await tr.predict());
            this.updateAxis((now - last) / 1000); last = now;
            this.drawPlot();
            this.stepsEl.textContent = U.fmtInt(tr.step);
            this.timeEl.textContent = ((now - t0) / 1000).toFixed(1);
            this.psnrEl.textContent = `${U.psnr(tr.lastLoss).toFixed(1)} dB`;
          },
        });
        // keep easing the axis after training stops
        c.loop((now, dt) => { if (tr.step >= MAX_STEPS) { this.updateAxis(dt); this.drawPlot(); } });
      });
    },
    async replay(c) {
      const ff = this.A.ff, fr = ff.frames;
      const t0 = performance.now();
      await c.play(9000, (e, raw) => {
        const step = Math.round(1000 * raw * raw);
        let k = 0;
        while (k + 1 < fr.length && fr[k + 1].step <= step) k++;
        this.drawImg(this.gF, this.ffFrames[k]);
        let n = 0;
        while (n < this.A.ffLossSteps.length && this.A.ffLossSteps[n] <= step) n++;
        this.hist = { steps: this.A.ffLossSteps.slice(0, n), loss: ff.loss.slice(0, n) };
        this.updateAxis(1 / 60); this.drawPlot();
        this.stepsEl.textContent = U.fmtInt(step);
        this.timeEl.textContent = ((performance.now() - t0) / 1000).toFixed(1);
        if (n) this.psnrEl.textContent = `${U.psnr(ff.loss[n - 1]).toFixed(1)} dB`;
      }, U.Ease.linear);
    },
    leave() { if (this.stop) this.stop(); this.dispose(); },
    dispose() { if (this.tr) { this.tr.dispose(); this.tr = null; } },
  });
})();
