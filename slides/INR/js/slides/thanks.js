/* Thank you: the title animation once more; press W to train an INR on the webcam, live */
(function () {
  const C = 600, R = 128;

  Deck.register('s-thanks', {
    keys: ['w', 'W'],
    init(ctx) {
      this.cv = ctx.$('#th-canvas');
      this.cap = ctx.$('.th-cap');
      this.hero = ASSETS.media.hero;
      this.ready = Promise.all(this.hero.frames.map(f => U.loadImage(f.src))).then(ims => { this.imgs = ims; });
      this.grab = document.createElement('canvas'); this.grab.width = R; this.grab.height = R;
      this.off = document.createElement('canvas');
      this.resize();
    },
    resize() { this.g = U.hidpi(this.cv, C, C); },
    draw(k, u) {
      const g = this.g, ims = this.imgs;
      g.globalAlpha = 1; g.drawImage(ims[k], 0, 0, C, C);
      if (u > 0 && k + 1 < ims.length) { g.globalAlpha = u; g.drawImage(ims[k + 1], 0, 0, C, C); g.globalAlpha = 1; }
      const f = this.hero.frames[u > 0.5 && k + 1 < ims.length ? k + 1 : k];
      this.cap.textContent = `an INR after ${f.step} training steps · ${f.psnr.toFixed(1)} dB`;
    },
    heroLoop(ctx) {
      this.loopCtx && this.loopCtx.cancel();
      const lc = this.loopCtx = ctx.child();
      lc.run(async c => {
        await this.ready;
        const n = this.imgs.length;
        while (true) {
          for (let k = 0; k < n - 1; k++) await c.play(k < 4 ? 260 : 420, u => this.draw(k, u), U.Ease.inOutSine);
          this.draw(n - 1, 0);
          await c.wait(5000);
        }
      });
    },
    enter(ctx) { this.heroLoop(ctx); },
    key(ctx, k) { if (this.cam) this.stopCam(ctx); else this.startCam(ctx); },

    /* ---- webcam INR ---- */
    sample() {
      const v = this.video, g = this.grab.getContext('2d', { willReadFrequently: true });
      const s = Math.min(v.videoWidth, v.videoHeight);
      g.save(); g.translate(R, 0); g.scale(-1, 1);                 // mirror, like a selfie view
      g.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, R, R);
      g.restore();
      const d = g.getImageData(0, 0, R, R).data, out = new Float32Array(R * R * 3);
      for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { out[j] = d[i] / 255; out[j + 1] = d[i + 1] / 255; out[j + 2] = d[i + 2] / 255; }
      return out;
    },
    async startCam(ctx) {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('no camera API');
        const backend = await INR.init();
        if (!backend || backend === 'cpu') throw new Error('no GPU backend');
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        const v = this.video = document.createElement('video');
        v.srcObject = stream; v.muted = true; v.playsInline = true;
        await v.play();
        this.cam = { stream };
        this.loopCtx && this.loopCtx.cancel();
        const model = new INR.Model({ inDim: 2, outDim: 3, features: 'fourier', m: 128, sigma: 10, width: 128, depth: 2, seed: 3 });
        const tr = this.tr = new INR.ImageFit(model, this.sample(), R, 3e-3);
        tr.stepsPerChunk = 4;
        this.cap.textContent = 'some random network';
        const camCtx = this.camCtx = ctx.child();
        this.stop = INR.run(camCtx, [tr], {
          targetMs: 45, maxSteps: 16,
          onFrame: async () => {
            const rgb = await tr.predict();
            U.putRGB(this.off, rgb, R);
            const g = this.g;
            g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
            g.drawImage(this.off, 0, 0, C, C);
            // live camera inset
            g.save(); U.roundRect(g, C - 186, C - 186, 168, 168, 12); g.clip();
            g.drawImage(this.grab, C - 186, C - 186, 168, 168);
            g.restore();
            g.strokeStyle = '#fff'; g.lineWidth = 3; U.roundRect(g, C - 186, C - 186, 168, 168, 12); g.stroke();
            this.cap.textContent = `some random network · step ${U.fmtInt(tr.step)} · ${U.psnr(tr.lastLoss).toFixed(1)} dB`;
            tr.setTarget(this.sample());
          },
        });
      } catch (e) {
        console.warn('webcam INR unavailable:', e);
        this.cap.textContent = 'webcam not available';
        this.cam = null;
      }
    },
    stopCam(ctx) {
      if (this.stop) { this.stop(); this.stop = null; }
      this.camCtx && this.camCtx.cancel();
      if (this.tr) { this.tr.dispose(); this.tr = null; }
      if (this.cam) { this.cam.stream.getTracks().forEach(t => t.stop()); this.cam = null; }
      if (this.video) { this.video.srcObject = null; this.video = null; }
      if (ctx) this.heroLoop(ctx);
    },
    leave() { this.stopCam(null); },
  });
})();
