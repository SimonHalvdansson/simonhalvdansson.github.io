/* Title: an INR (Fourier features + MLP) learning a photo, replayed from precomputed snapshots */
Deck.register('s-title', {
  init(ctx) {
    this.cv = ctx.$('#title-canvas');
    this.readout = ctx.$('.t-readout');
    this.hero = ASSETS.media.hero;
    this.imgs = [];
    this.ready = Promise.all(this.hero.frames.map(f => U.loadImage(f.src))).then(ims => { this.imgs = ims; });
    this.size();
    ctx.$('.t-probe').style.display = 'none';
  },
  size() { this.g = U.hidpi(this.cv, 660, 660); },
  resize() { this.size(); this.draw(this.lastK || 0, this.lastU || 0); },
  draw(k, u) {
    const g = this.g, ims = this.imgs;
    if (!ims.length) return;
    this.lastK = k; this.lastU = u;
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.globalAlpha = 1;
    g.drawImage(ims[k], 0, 0, 660, 660);
    if (u > 0 && k + 1 < ims.length) { g.globalAlpha = u; g.drawImage(ims[k + 1], 0, 0, 660, 660); g.globalAlpha = 1; }
    const f = this.hero.frames[u > 0.5 && k + 1 < ims.length ? k + 1 : k];
    this.readout.textContent = `training step ${String(f.step).padStart(4, ' ')}  ·  PSNR ${f.psnr.toFixed(1)} dB`;
  },
  enter(ctx) {
    ctx.run(async c => {
      await this.ready;
      const n = this.imgs.length;
      while (true) {
        for (let k = 0; k < n - 1; k++) {
          const dur = k < 4 ? 260 : 420;
          await c.play(dur, u => this.draw(k, u), U.Ease.inOutSine);
        }
        this.draw(n - 1, 0);
        await c.wait(4200);
        await c.play(900, u => {           // fade back to the start
          const g = this.g;
          g.globalAlpha = 1; g.drawImage(this.imgs[n - 1], 0, 0, 660, 660);
          g.globalAlpha = u; g.drawImage(this.imgs[0], 0, 0, 660, 660); g.globalAlpha = 1;
        });
        this.draw(0, 0);
        await c.wait(600);
      }
    });
  },
});
