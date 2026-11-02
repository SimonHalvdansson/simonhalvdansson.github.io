/* plot.js — tiny canvas plotting helper (linear / log axes, lines, labels) */
(function () {
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const pow10Label = e => (e === 0 ? '1' : e === 1 ? '10' : '10' + String(e).split('').map(c => SUP[c]).join(''));

  function niceStep(range, target) {
    const raw = range / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    const s = n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10;
    return s * mag;
  }

  class Plot {
    /* opts: { w, h, pad:{l,r,t,b}, x:{min,max,log,label,ticks,fmt,grid}, y:{...}, font } */
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.o = Object.assign({ pad: { l: 90, r: 30, t: 20, b: 70 }, font: 22, axes: true }, opts);
      this.o.x = Object.assign({ min: 0, max: 1, log: false, ticks: 5, grid: true }, opts.x || {});
      this.o.y = Object.assign({ min: 0, max: 1, log: false, ticks: 5, grid: true }, opts.y || {});
      this.resize();
    }
    resize() {
      this.ctx = U.hidpi(this.canvas, this.o.w, this.o.h);
      const p = this.o.pad;
      this.x0 = p.l; this.x1 = this.o.w - p.r;
      this.y0 = this.o.h - p.b; this.y1 = p.t;
    }
    setX(min, max) { this.o.x.min = min; this.o.x.max = max; }
    setY(min, max) { this.o.y.min = min; this.o.y.max = max; }
    sx(x) {
      const a = this.o.x;
      const t = a.log ? (Math.log10(Math.max(x, 1e-30)) - Math.log10(a.min)) / (Math.log10(a.max) - Math.log10(a.min))
                      : (x - a.min) / (a.max - a.min);
      return this.x0 + t * (this.x1 - this.x0);
    }
    sy(y) {
      const a = this.o.y;
      const t = a.log ? (Math.log10(Math.max(y, 1e-30)) - Math.log10(a.min)) / (Math.log10(a.max) - Math.log10(a.min))
                      : (y - a.min) / (a.max - a.min);
      return this.y0 + t * (this.y1 - this.y0);
    }
    clear(bg = null) {
      const c = this.ctx;
      if (bg) { c.fillStyle = bg; c.fillRect(0, 0, this.o.w, this.o.h); }
      else c.clearRect(0, 0, this.o.w, this.o.h);
    }
    ticks(a) {
      if (a.tickValues) return a.tickValues;
      if (a.log) {
        const out = [];
        for (let e = Math.ceil(Math.log10(a.min) - 1e-9); e <= Math.floor(Math.log10(a.max) + 1e-9); e++) out.push(Math.pow(10, e));
        if (a.tickEvery) return out.filter((v, i) => i % a.tickEvery === 0);
        return out;
      }
      const st = niceStep(a.max - a.min, a.ticks);
      const out = [];
      for (let v = Math.ceil(a.min / st) * st; v <= a.max + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
      return out;
    }
    fmt(a, v) {
      if (a.fmt) return a.fmt(v);
      if (a.log) return pow10Label(Math.round(Math.log10(v)));
      if (Math.abs(v) >= 1000) return U.fmtCompact(v);
      return String(+v.toFixed(6));
    }
    drawAxes(opts = {}) {
      const c = this.ctx, o = this.o;
      c.save();
      c.font = `400 ${o.font}px "Fira Sans", sans-serif`;
      c.lineWidth = 1;
      // grid
      const xt = this.ticks(o.x), yt = this.ticks(o.y);
      c.strokeStyle = '#e3e1d9';
      if (o.y.grid) for (const v of yt) { const y = Math.round(this.sy(v)) + 0.5; c.beginPath(); c.moveTo(this.x0, y); c.lineTo(this.x1, y); c.stroke(); }
      if (o.x.grid) for (const v of xt) { const x = Math.round(this.sx(v)) + 0.5; c.beginPath(); c.moveTo(x, this.y0); c.lineTo(x, this.y1); c.stroke(); }
      // axes
      c.strokeStyle = '#c3c2b7'; c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(this.x0, this.y1); c.lineTo(this.x0, this.y0); c.lineTo(this.x1, this.y0); c.stroke();
      // tick labels
      c.fillStyle = '#8a8780';
      c.textAlign = 'center'; c.textBaseline = 'top';
      if (!o.x.hideTicks) for (const v of xt) c.fillText(this.fmt(o.x, v), this.sx(v), this.y0 + 10);
      c.textAlign = 'right'; c.textBaseline = 'middle';
      if (!o.y.hideTicks) for (const v of yt) c.fillText(this.fmt(o.y, v), this.x0 - 12, this.sy(v));
      // axis titles
      c.fillStyle = '#4b4f5c';
      c.font = `500 ${o.font}px "Fira Sans", sans-serif`;
      if (o.x.label) { c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText(o.x.label, (this.x0 + this.x1) / 2, o.h - 2); }
      if (o.y.label) {
        c.save(); c.translate(o.y.labelX ?? 18, (this.y0 + this.y1) / 2); c.rotate(-Math.PI / 2);
        c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText(o.y.label, 0, 0); c.restore();
      }
      c.restore();
    }
    /* draw a polyline from arrays xs, ys (data space). opts: color, width, alpha, upto (index), clip */
    line(xs, ys, opts = {}) {
      const c = this.ctx;
      const n = Math.min(xs.length, ys.length, opts.upto ?? Infinity);
      if (n < 2) return;
      c.save();
      if (opts.clip !== false) { c.beginPath(); c.rect(this.x0, this.y1 - 4, this.x1 - this.x0 + 4, this.y0 - this.y1 + 8); c.clip(); }
      c.strokeStyle = opts.color || '#2a78d6';
      c.globalAlpha = opts.alpha ?? 1;
      c.lineWidth = opts.width || 3;
      c.lineJoin = 'round'; c.lineCap = 'round';
      c.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const y = ys[i];
        if (!isFinite(y)) { started = false; continue; }
        const px = this.sx(xs[i]), py = this.sy(y);
        if (!started) { c.moveTo(px, py); started = true; } else c.lineTo(px, py);
      }
      c.stroke();
      c.restore();
    }
    /* fill between curve and baseline value */
    area(xs, ys, base, opts = {}) {
      const c = this.ctx, n = Math.min(xs.length, ys.length);
      c.save();
      c.fillStyle = opts.color || 'rgba(42,120,214,0.1)';
      c.beginPath();
      c.moveTo(this.sx(xs[0]), this.sy(base));
      for (let i = 0; i < n; i++) c.lineTo(this.sx(xs[i]), this.sy(ys[i]));
      c.lineTo(this.sx(xs[n - 1]), this.sy(base));
      c.closePath(); c.fill();
      c.restore();
    }
    dot(x, y, opts = {}) {
      const c = this.ctx, px = this.sx(x), py = this.sy(y), r = opts.r || 6;
      c.save();
      c.fillStyle = opts.ring || '#fbfaf6';
      c.beginPath(); c.arc(px, py, r + 2.5, 0, Math.PI * 2); c.fill();
      c.fillStyle = opts.color || '#2a78d6';
      c.beginPath(); c.arc(px, py, r, 0, Math.PI * 2); c.fill();
      c.restore();
    }
    text(str, px, py, opts = {}) {
      const c = this.ctx;
      c.save();
      c.font = opts.font || `500 ${this.o.font}px "Fira Sans", sans-serif`;
      c.fillStyle = opts.color || '#1d2130';
      c.textAlign = opts.align || 'left';
      c.textBaseline = opts.baseline || 'middle';
      if (opts.halo) { c.lineWidth = 6; c.strokeStyle = opts.halo; c.lineJoin = 'round'; c.strokeText(str, px, py); }
      c.fillText(str, px, py);
      c.restore();
    }
    /* end-of-line label with a short colored key line */
    endLabel(str, x, y, color, opts = {}) {
      const px = this.sx(x) + (opts.dx ?? 14), py = this.sy(y) + (opts.dy ?? 0);
      const c = this.ctx;
      c.save();
      c.strokeStyle = color; c.lineWidth = 4; c.lineCap = 'round';
      c.beginPath(); c.moveTo(px, py); c.lineTo(px + 18, py); c.stroke();
      c.restore();
      this.text(str, px + 26, py, { color: '#1d2130', font: opts.font, halo: '#fbfaf6' });
    }
  }

  window.Plot = Plot;
  Plot.pow10Label = pow10Label;
})();
