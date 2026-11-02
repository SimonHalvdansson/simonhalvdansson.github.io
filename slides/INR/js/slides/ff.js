/* Fourier features: gamma(x) = [cos(2 pi B x), sin(2 pi B x)] — random plane waves fed to an MLP */
(function () {
  const T = 110, NT = 8, RES = 96;
  const gauss = U.gaussian(21);
  const feats = Array.from({ length: NT }, (_, k) => ({ b: [gauss() * 3.2, gauss() * 3.2], sin: k % 2 === 1 }));
  const BG = U.hexToRgb('#f3f1ea'), BL = U.hexToRgb('#2a78d6');

  Deck.register('s-ff', {
    steps: 2,
    init(ctx) {
      const box = ctx.$('.ff-tiles');
      this.tiles = feats.map(f => {
        const c = U.el('canvas', {}, box);
        // render the plane wave once at RES x RES
        c.width = RES; c.height = RES;
        const g = c.getContext('2d'), im = g.createImageData(RES, RES);
        for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
          const x = i / (RES - 1), y = j / (RES - 1);
          const ph = 2 * Math.PI * (f.b[0] * x + f.b[1] * y);
          const v = (f.sin ? Math.sin(ph) : Math.cos(ph)) * 0.5 + 0.5;
          const k = (j * RES + i) * 4;
          for (let ch = 0; ch < 3; ch++) im.data[k + ch] = BG[ch] + (BL[ch] - BG[ch]) * v;
          im.data[k + 3] = 255;
        }
        g.putImageData(im, 0, 0);
        const base = document.createElement('canvas'); base.width = RES; base.height = RES;
        base.getContext('2d').drawImage(c, 0, 0);
        return { c, base, f };
      });
      this.net = new MLPDiagram(ctx.$('#ff-net'), { layers: [7, 6, 6, 3], x: 24, y: 10, w: 300, h: 240, r: 13, ellipsis: [0] });
      const oc = ['#d64545', '#2e9e4a', '#2a78d6'];
      this.net.nodes[3].forEach((n, i) => n.setAttribute('stroke', oc[i]));
      this.out = ctx.$('#ff-out');
      const fr = ASSETS.relu.ff.frames;
      U.loadImage(fr[fr.length - 1].src).then(im => { this.outImg = im; this.drawOut(); });
      this.probe = { u: 0.4, v: 0.4 };
      this.resize();
    },
    resize() {
      this.tiles.forEach(t => { t.g = U.hidpi(t.c, T, T); });
      this.go = U.hidpi(this.out, 230, 230);
      this.drawTiles(); this.drawOut();
    },
    drawTiles() {
      const { u, v } = this.probe;
      for (const t of this.tiles) {
        const g = t.g;
        g.imageSmoothingEnabled = true;
        g.drawImage(t.base, 0, 0, T, T);
        g.fillStyle = '#1d2130';
        g.beginPath(); g.arc(u * T, v * T, 5, 0, 2 * Math.PI); g.fill();
        g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
      }
    },
    drawOut() {
      if (!this.outImg) return;
      const g = this.go, { u, v } = this.probe;
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(this.outImg, 0, 0, 230, 230);
      g.strokeStyle = '#fff'; g.lineWidth = 3;
      g.beginPath(); g.arc(u * 230, v * 230, 10, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = 'rgba(0,0,0,.5)'; g.lineWidth = 1.2;
      g.beginPath(); g.arc(u * 230, v * 230, 12, 0, 2 * Math.PI); g.stroke();
    },
    enter(ctx) {
      this.drawTiles(); this.drawOut();
      const pts = [[0.4, 0.4], [0.7, 0.3], [0.25, 0.72], [0.62, 0.66], [0.15, 0.3], [0.82, 0.78]];
      ctx.run(async c => {
        let k = 1;
        await c.wait(400);
        while (true) {
          const [tu, tv] = pts[k % pts.length]; k++;
          const fu = this.probe.u, fv = this.probe.v;
          await c.play(1000, t => { this.probe.u = U.lerp(fu, tu, t); this.probe.v = U.lerp(fv, tv, t); this.drawTiles(); this.drawOut(); });
          await this.net.pulse(c, { dur: 360, color: '#2a78d6' });
          await c.wait(900);
        }
      });
    },
  });
})();
