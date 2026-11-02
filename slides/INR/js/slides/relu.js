/* Attempt 1: plain ReLU MLPs of four sizes learning a 128x128 image (precomputed with PyTorch, replayed) */
(function () {
  const RAMP = ['#f2a072', '#eb6834', '#c04a17', '#8a3008'];
  const TILE = 300;
  const PLAY_MS = 15000;

  Deck.register('s-relu', {
    init(ctx) {
      const A = this.A = ASSETS.relu;
      const tiles = ctx.$('.relu-tiles');
      this.tiles = A.runs.map((r, i) => {
        const fig = U.el('figure', { class: 'tile' }, tiles);
        const cv = U.el('canvas', { class: 'tile-c' }, fig);
        const cap = U.el('figcaption', {}, fig,
          `<b><i class="swatch" style="background:${RAMP[i]};width:24px;height:7px"></i>${r.width} × ${r.depth}</b>` +
          `<span>${U.fmtCompact(r.params)} params · <span class="psnr">—</span></span>`);
        return { fig, cv, psnr: cap.querySelector('.psnr'), imgs: [] };
      });
      this.targetCv = ctx.$('.relu-tiles .target canvas');
      this.stepEl = ctx.$('.relu-step span');
      this.plot = new Plot(ctx.$('#relu-plot'), {
        w: 1160, h: 340, pad: { l: 96, r: 96, t: 16, b: 62 }, font: 22,
        x: { min: 0, max: A.steps, tickValues: [0, 1000, 2000, 3000, 4000, 5000], label: 'training step', grid: false },
        y: { min: 1e-3, max: 1e-1, log: true, label: 'Mean squared error (MSE)', labelX: 6 },
      });
      // decode snapshot images
      this.ready = Promise.all([
        U.loadImage(A.target).then(im => { this.target = im; }),
        ...A.runs.map((r, i) => Promise.all(r.frames.map(f => U.loadImage(f.src))).then(ims => { this.tiles[i].imgs = ims; })),
      ]);
      this.resize();
    },
    resize() {
      this.gt = U.hidpi(this.targetCv, TILE, TILE);
      this.tiles.forEach(t => { t.g = U.hidpi(t.cv, TILE, TILE); });
      this.plot.resize();
      if (this.lastStep !== undefined) this.render(this.lastStep);
    },
    drawImg(g, im, alpha = 1) {
      g.save(); U.roundRect(g, 0, 0, TILE, TILE, 14); g.clip();
      g.globalAlpha = alpha; g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(im, 0, 0, TILE, TILE); g.restore();
    },
    render(step) {
      this.lastStep = step;
      const A = this.A;
      if (this.target) this.drawImg(this.gt, this.target);
      A.runs.forEach((r, i) => {
        const t = this.tiles[i];
        if (!t.imgs.length) return;
        const fr = r.frames;
        let k = 0;
        while (k + 1 < fr.length && fr[k + 1].step <= step) k++;
        const k2 = Math.min(fr.length - 1, k + 1);
        const w = k2 === k ? 0 : U.clamp((step - fr[k].step) / (fr[k2].step - fr[k].step), 0, 1);
        this.drawImg(t.g, t.imgs[k]);
        if (w > 0) this.drawImg(t.g, t.imgs[k2], w);
        const mse = U.lerp(fr[k].mse, fr[k2].mse, w);
        t.psnr.textContent = step === 0 ? '—' : `${U.psnr(mse).toFixed(1)} dB`;
      });
      // loss curves
      const p = this.plot;
      p.clear(); p.drawAxes();
      let n = 0;
      while (n < A.lossSteps.length && A.lossSteps[n] <= step) n++;
      A.runs.forEach((r, i) => p.line(A.lossSteps, r.loss, { color: RAMP[i], width: 2.6, upto: n }));
      if (n > 1) A.runs.forEach((r, i) => p.dot(A.lossSteps[n - 1], r.loss[n - 1], { r: 5.5, color: RAMP[i] }));
      if (step >= A.steps) {
        A.runs.forEach((r, i) => p.endLabel(U.fmtCompact(r.params), A.steps, r.loss[r.loss.length - 1], RAMP[i], { dx: 16 }));
      }
      this.stepEl.textContent = U.fmtInt(step);
    },
    enter(ctx, step) {
      ctx.run(async c => {
        await this.ready;
        if (step > 0) { this.render(this.A.steps); return; }
        this.render(0);
        await c.wait(700);
        await c.play(PLAY_MS, (e, raw) => this.render(Math.round(this.A.steps * raw * raw)), U.Ease.linear);
        this.render(this.A.steps);
      });
    },
    step(ctx, step) {
      if (step >= 1) {
        ctx.cancel();
        ctx.run(async () => { await this.ready; this.render(this.A.steps); });
      }
    },
  });
})();
