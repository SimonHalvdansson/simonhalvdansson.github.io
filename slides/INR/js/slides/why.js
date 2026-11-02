/* Why are INRs useful? four small looping illustrations */
(function () {
  const W = 400, H = 300;
  /* draw base text with a subscript, centred at x */
  function subText(g, base, sub, x, y, size, color, font = 'KaTeX_Math, serif') {
    g.save();
    g.fillStyle = color; g.textAlign = 'left';
    g.font = `italic ${size}px ${font}`;
    const wb = g.measureText(base).width;
    g.font = `italic ${size * 0.68}px ${font}`;
    const ws = g.measureText(sub).width;
    const x0 = x - (wb + ws) / 2;
    g.font = `italic ${size}px ${font}`; g.fillText(base, x0, y);
    g.font = `italic ${size * 0.68}px ${font}`; g.fillText(sub, x0 + wb + 1, y + size * 0.22);
    g.restore();
  }
  const BLUE = '#2a78d6', INK = '#1d2130', GRAY = '#b9b6ad', BG = '#f6f5f0';

  /* 1. compact: a grid of pixels collapses into a small network */
  function drawCompact(st, g, t) {
    g.fillStyle = BG; g.fillRect(0, 0, W, H);
    if (!st.px) return;
    const n = 12, cell = 16, gx = 34, gy = 44;
    const cyc = (t % 6) / 6;                      // 0..1
    const u = U.Ease.inOutCubic(U.seg(cyc, 0.15, 0.5)) * (1 - U.Ease.inOutCubic(U.seg(cyc, 0.75, 0.95)));
    const nx = 300, ny = 140;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const c = st.px[(j * n + i)];
      const x0 = gx + i * cell, y0 = gy + j * cell;
      const x = U.lerp(x0, nx - 6 + (i - n / 2) * 1.2, u), y = U.lerp(y0, ny - 6 + (j - n / 2) * 1.2, u);
      const s = U.lerp(cell - 1.5, 3, u);
      g.globalAlpha = 1 - 0.85 * u;
      g.fillStyle = c; g.fillRect(x, y, s, s);
    }
    g.globalAlpha = 1;
    // network icon
    const a = U.seg(u, 0.6, 1);
    g.globalAlpha = 0.25 + 0.75 * a;
    const L = [[nx - 34, [ny - 18, ny + 18]], [nx, [ny - 30, ny, ny + 30]], [nx + 34, [ny - 12, ny + 12]]];
    g.strokeStyle = '#cfccc3'; g.lineWidth = 1.3;
    for (let l = 0; l < 2; l++) for (const ya of L[l][1]) for (const yb of L[l + 1][1]) { g.beginPath(); g.moveTo(L[l][0], ya); g.lineTo(L[l + 1][0], yb); g.stroke(); }
    for (const [x, ysL] of L) for (const y of ysL) { g.fillStyle = '#fff'; g.strokeStyle = BLUE; g.lineWidth = 2.2; g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); g.stroke(); }
    g.globalAlpha = 1;
    g.font = '500 20px "Fira Sans"'; g.fillStyle = '#4b4f5c'; g.textAlign = 'center';
    g.fillText('samples', gx + n * cell / 2, gy + n * cell + 30);
    g.fillStyle = BLUE; g.fillText('θ', nx, ny + 64);
    g.textAlign = 'left';
  }

  /* 2. continuous: one function, sampled at any resolution */
  const fC = x => 0.5 + 0.28 * Math.sin(2 * Math.PI * (1.3 * x + 0.1)) + 0.12 * Math.sin(2 * Math.PI * 4.1 * x);
  function drawContinuous(g, t) {
    g.fillStyle = BG; g.fillRect(0, 0, W, H);
    const x0 = 30, x1 = W - 30, y0 = H - 46, y1 = 30;
    const X = x => x0 + x * (x1 - x0), Y = v => y0 + v * (y1 - y0);
    g.strokeStyle = GRAY; g.lineWidth = 5; g.lineJoin = 'round';
    g.beginPath(); for (let i = 0; i <= 200; i++) { const x = i / 200; i ? g.lineTo(X(x), Y(fC(x))) : g.moveTo(X(x), Y(fC(x))); } g.stroke();
    const levels = [6, 12, 24, 60];
    const k = Math.floor(t / 1.6) % levels.length, n = levels[k];
    g.fillStyle = BLUE;
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) / n;
      g.beginPath(); g.arc(X(x), Y(fC(x)), n > 30 ? 3.2 : 5.5, 0, 7); g.fill();
    }
    g.font = '500 20px "Fira Sans"'; g.fillStyle = '#4b4f5c'; g.textAlign = 'center';
    g.fillText(`${n} samples of the same function`, W / 2, H - 14);
    g.textAlign = 'left';
  }

  /* 3. differentiable: a moving tangent */
  function drawDiff(g, t) {
    g.fillStyle = BG; g.fillRect(0, 0, W, H);
    const x0 = 30, x1 = W - 30, y0 = H - 46, y1 = 26;
    const X = x => x0 + x * (x1 - x0), Y = v => y0 + v * (y1 - y0);
    g.strokeStyle = GRAY; g.lineWidth = 5;
    g.beginPath(); for (let i = 0; i <= 200; i++) { const x = i / 200; i ? g.lineTo(X(x), Y(fC(x))) : g.moveTo(X(x), Y(fC(x))); } g.stroke();
    const xp = 0.5 + 0.4 * Math.sin(t * 0.7);
    const h = 1e-4, d = (fC(xp + h) - fC(xp - h)) / (2 * h);
    const sx = (x1 - x0), sy = (y1 - y0);
    const ang = Math.atan2(d * sy, sx), len = 80;
    const px = X(xp), py = Y(fC(xp));
    g.strokeStyle = BLUE; g.lineWidth = 3.5; g.lineCap = 'round';
    g.beginPath(); g.moveTo(px - Math.cos(ang) * len, py - Math.sin(ang) * len); g.lineTo(px + Math.cos(ang) * len, py + Math.sin(ang) * len); g.stroke();
    g.fillStyle = BLUE; g.beginPath(); g.arc(px, py, 7, 0, 7); g.fill();
    g.font = '22px KaTeX_Main, serif'; g.fillStyle = INK; g.textAlign = 'center';
    g.fillText(`∂f/∂x = ${d >= 0 ? ' ' : '−'}${Math.abs(d).toFixed(2)}`, W / 2, H - 14);
    g.textAlign = 'left';
  }

  /* 4. physics: f_theta -> A -> y, gradient back */
  function drawPhys(g, t) {
    g.fillStyle = BG; g.fillRect(0, 0, W, H);
    const bx = [[30, 'f', BLUE], [165, 'A', '#b07800'], [300, 'y', INK]];
    const by = 96, bw = 74, bh = 74;
    for (const [x, s, c] of bx) {
      g.fillStyle = '#fff'; g.strokeStyle = U.rgba(c.length === 7 ? c : '#1d2130', 0.5); g.lineWidth = 2;
      U.roundRect(g, x, by, bw, bh, 14); g.fill(); g.stroke();
      if (s === 'f') subText(g, 'f', 'θ', x + bw / 2, by + bh / 2 + 11, 32, c);
      else { g.fillStyle = c; g.font = 'italic 32px KaTeX_Math, serif'; g.textAlign = 'center'; g.fillText(s, x + bw / 2, by + bh / 2 + 11); }
    }
    U.arrow(g, 30 + bw + 6, by + 26, 165 - 8, by + 26, 11, 2.5, '#4b4f5c');
    U.arrow(g, 165 + bw + 6, by + 26, 300 - 8, by + 26, 11, 2.5, '#4b4f5c');
    // gradient arc back from y to f_theta, ending in an arrowhead aligned with the curve
    const yB = by + bh + 6, tipY = yB, endY = yB + 16;
    g.strokeStyle = '#d64545'; g.lineWidth = 2.5; g.lineCap = 'round';
    g.beginPath(); g.moveTo(337, yB); g.bezierCurveTo(337, 252, 67, 252, 67, endY); g.stroke();
    g.fillStyle = '#d64545';
    g.beginPath(); g.moveTo(67, tipY); g.lineTo(60, endY + 2); g.lineTo(74, endY + 2); g.closePath(); g.fill();
    // moving pulses
    const u = (t % 2.4) / 2.4;
    if (u < 0.5) {
      const v = u / 0.5, x = U.lerp(30 + bw, 300, v);
      g.fillStyle = BLUE; g.beginPath(); g.arc(x, by + 26, 6, 0, 7); g.fill();
    } else {
      const v = (u - 0.5) / 0.5;
      const bez = (a, b, c, d, s) => (1 - s) ** 3 * a + 3 * (1 - s) ** 2 * s * b + 3 * (1 - s) * s * s * c + s ** 3 * d;
      const x = bez(337, 337, 67, 67, v), y = bez(by + bh + 6, 252, 252, by + bh + 22, v);
      g.fillStyle = '#d64545'; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill();
    }
    g.font = '500 19px "Fira Sans"'; g.fillStyle = '#d64545'; g.textAlign = 'center';
    g.fillText('∇θ', W / 2, 272);
    g.textAlign = 'left';
  }

  Deck.register('s-why', {
    init(ctx) {
      this.cv = Object.fromEntries(ctx.$$('.why-card').map(c => [c.dataset.k, c.querySelector('canvas')]));
      this.st = {};
      U.loadImage(ASSETS.media.storefront128).then(im => {
        const c = document.createElement('canvas'); c.width = 12; c.height = 12;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(im, 0, 0, 12, 12);
        const d = g.getImageData(0, 0, 12, 12).data;
        this.st.px = Array.from({ length: 144 }, (_, i) => `rgb(${d[4 * i]},${d[4 * i + 1]},${d[4 * i + 2]})`);
      });
      this.resize();
    },
    resize() { this.g = Object.fromEntries(Object.entries(this.cv).map(([k, c]) => [k, U.hidpi(c, W, H)])); },
    enter(ctx) {
      const t0 = performance.now() / 1000;
      ctx.loop(now => {
        const t = Math.max(0, now - t0);
        drawContinuous(this.g.continuous, t);
        drawDiff(this.g.diff, t);
        drawPhys(this.g.physics, t);
        drawCompact(this.st, this.g.compact, t);
      });
    },
  });
})();
