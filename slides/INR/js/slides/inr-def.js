/* INR definition: explicit pixel grid vs. implicit function f_theta(x, y) = (r, g, b) */
(function () {
  const IMG = 400, ZOOM = 270, ZN = 5, ZX = 430, ZY = 50;          // main image size, zoom inset size, zoom cells per side
  const N = 128;

  Deck.register('s-inr', {
    init(ctx) {
      this.ci = ctx.$('#inr-img');
      this.cz = ctx.$('#inr-zoom');
      this.svg = ctx.$('#inr-net');
      this.link = U.svg('svg', { width: 700, height: 440, style: 'position:absolute;left:0;top:0;overflow:visible;pointer-events:none' }, ctx.$('.inr-imgwrap'));
      this.l1 = U.svg('line', { stroke: '#1d2130', 'stroke-width': 1.5, opacity: 0.45 }, this.link);
      this.l2 = U.svg('line', { stroke: '#1d2130', 'stroke-width': 1.5, opacity: 0.45 }, this.link);
      this.probe = { u: 0.37, v: 0.33 };
      U.loadImage(ASSETS.media.storefront128).then(im => {
        const c = document.createElement('canvas'); c.width = N; c.height = N;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(im, 0, 0);
        this.data = g.getImageData(0, 0, N, N).data;
        this.img = c;
        this.draw();
      });
      // network diagram
      const net = this.net = new MLPDiagram(this.svg, {
        layers: [2, 7, 7, 7, 3], x: 190, y: 40, w: 520, h: 400, r: 17, ellipsis: [1, 2, 3],
      });
      net.label(0, 0, 'x', { size: 24 }); net.label(0, 1, 'y', { size: 24 });
      const oc = ['#d64545', '#2e9e4a', '#2a78d6'];
      ['r', 'g', 'b'].forEach((s, i) => {
        net.nodes[4][i].setAttribute('stroke', oc[i]);
        net.label(4, i, s, { size: 24, color: oc[i] });
      });
      const T = (x, y, anchor, size = 26, fill = '#1d2130') => {
        const t = U.svg('text', { x, y, 'text-anchor': anchor, 'font-size': size, fill, 'font-family': 'KaTeX_Main, serif' }, this.svg);
        return t;
      };
      const pin = net.posVisible[0], pout = net.posVisible[4];
      this.tIn = [T(pin[0][0] - 34, pin[0][1] + 9, 'end'), T(pin[1][0] - 34, pin[1][1] + 9, 'end')];
      this.tOut = pout.map(p => T(p[0] + 34, p[1] + 9, 'start'));
      this.sw = U.svg('rect', { x: pout[0][0] + 120, y: pout[1][1] - 44, width: 88, height: 88, rx: 14, fill: '#ccc', stroke: 'rgba(0,0,0,.12)' }, this.svg);
      const cap = U.svg('text', { x: 450, y: 500, 'text-anchor': 'middle', 'font-size': 26, fill: '#8a8780', 'font-family': 'Fira Sans' }, this.svg);
      cap.textContent = 'all weights and biases: θ';
      this.resize();
    },
    resize() {
      this.gi = U.hidpi(this.ci, IMG, IMG);
      this.gz = U.hidpi(this.cz, ZOOM, ZOOM);
      this.draw();
    },
    color(px, py) {
      const k = (py * N + px) * 4, d = this.data;
      return [d[k], d[k + 1], d[k + 2]];
    },
    draw() {
      if (!this.data) return;
      const gi = this.gi, gz = this.gz;
      gi.imageSmoothingEnabled = false;
      gi.clearRect(0, 0, IMG, IMG);
      gi.save(); U.roundRect(gi, 0, 0, IMG, IMG, 14); gi.clip();
      gi.drawImage(this.img, 0, 0, IMG, IMG);
      gi.restore();
      const px = Math.round(this.probe.u * (N - 1)), py = Math.round(this.probe.v * (N - 1));
      const cs = IMG / N, h = Math.floor(ZN / 2);
      // probe box on main image
      gi.strokeStyle = '#fff'; gi.lineWidth = 3;
      gi.strokeRect((px - h) * cs, (py - h) * cs, ZN * cs, ZN * cs);
      gi.strokeStyle = '#1d2130'; gi.lineWidth = 1.2;
      gi.strokeRect((px - h) * cs - 2, (py - h) * cs - 2, ZN * cs + 4, ZN * cs + 4);
      const bx = (px - h + ZN) * cs + 2, by0 = (py - h) * cs - 2, by1 = (py - h + ZN) * cs + 2;
      this.l1.setAttribute('x1', bx); this.l1.setAttribute('y1', by0); this.l1.setAttribute('x2', ZX); this.l1.setAttribute('y2', ZY);
      this.l2.setAttribute('x1', bx); this.l2.setAttribute('y1', by1); this.l2.setAttribute('x2', ZX); this.l2.setAttribute('y2', ZY + ZOOM);
      // zoom inset: cells with their numbers
      const zc = ZOOM / ZN;
      gz.clearRect(0, 0, ZOOM, ZOOM);
      gz.save(); U.roundRect(gz, 0, 0, ZOOM, ZOOM, 14); gz.clip();
      for (let j = 0; j < ZN; j++) for (let i = 0; i < ZN; i++) {
        const qx = U.clamp(px - h + i, 0, N - 1), qy = U.clamp(py - h + j, 0, N - 1);
        const [r, g, b] = this.color(qx, qy);
        gz.fillStyle = `rgb(${r},${g},${b})`;
        gz.fillRect(i * zc, j * zc, zc + 0.5, zc + 0.5);
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        gz.fillStyle = lum > 140 ? 'rgba(29,33,48,0.8)' : 'rgba(255,255,255,0.88)';
        gz.font = '500 13.5px "Fira Mono", monospace'; gz.textAlign = 'center'; gz.textBaseline = 'middle';
        gz.fillText((r / 255).toFixed(2), i * zc + zc / 2, j * zc + zc / 2 - 15);
        gz.fillText((g / 255).toFixed(2), i * zc + zc / 2, j * zc + zc / 2);
        gz.fillText((b / 255).toFixed(2), i * zc + zc / 2, j * zc + zc / 2 + 15);
      }
      gz.strokeStyle = 'rgba(255,255,255,0.5)'; gz.lineWidth = 1;
      for (let i = 1; i < ZN; i++) {
        gz.beginPath(); gz.moveTo(i * zc, 0); gz.lineTo(i * zc, ZOOM); gz.stroke();
        gz.beginPath(); gz.moveTo(0, i * zc); gz.lineTo(ZOOM, i * zc); gz.stroke();
      }
      gz.strokeStyle = '#fff'; gz.lineWidth = 3.5; gz.strokeRect(h * zc + 1.5, h * zc + 1.5, zc - 3, zc - 3);
      gz.restore();
      // values in the network
      const [r, g, b] = this.color(px, py);
      this.tIn[0].textContent = (px / (N - 1)).toFixed(2);
      this.tIn[1].textContent = (py / (N - 1)).toFixed(2);
      if (this.showOut) {
        this.tOut[0].textContent = (r / 255).toFixed(2);
        this.tOut[1].textContent = (g / 255).toFixed(2);
        this.tOut[2].textContent = (b / 255).toFixed(2);
        this.sw.setAttribute('fill', `rgb(${r},${g},${b})`);
      }
    },
    enter(ctx) {
      this.showOut = true;
      this.draw();
      const targets = [[0.37, 0.33], [0.60, 0.48], [0.22, 0.70], [0.82, 0.30], [0.50, 0.62], [0.68, 0.80], [0.30, 0.52]];
      ctx.run(async c => {
        let k = 0;
        await c.wait(500);
        while (true) {
          const [tu, tv] = targets[k % targets.length]; k++;
          const fu = this.probe.u, fv = this.probe.v;
          this.showOut = false;
          this.tOut.forEach(t => { t.textContent = ''; });
          this.sw.setAttribute('fill', '#ebe8e0');
          await c.play(900, t => { this.probe.u = U.lerp(fu, tu, t); this.probe.v = U.lerp(fv, tv, t); this.draw(); });
          await this.net.pulse(c, { dur: 380, color: '#2a78d6' });
          this.showOut = true; this.draw();
          await c.wait(1700);
        }
      });
    },
  });
})();
